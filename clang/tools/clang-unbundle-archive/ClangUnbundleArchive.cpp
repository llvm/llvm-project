//===-- clang-unbundle-archive/ClangUnbundleArchive.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a clang-unbundle-archive that unbundles an archive
/// of offload-bundles and creates separate archives for each target that is
/// found in the bundles. This is only done for the offload targets, not the
/// host target.
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangUnbundleArchiveCategory("clang-unbundle-archive options");

static cl::opt<std::string>
    InputFileName("input", cl::Optional, cl::desc("[<input file>]"),
                  cl::cat(ClangUnbundleArchiveCategory));
static cl::opt<std::string> OutputFile("output", cl::Optional,
                                       cl::desc("[<output file>]"),
                                       cl::cat(ClangUnbundleArchiveCategory));
static cl::opt<std::string>
    HostOutputFile("host-output", cl::Optional,
                   cl::desc("[<host output file>]"),
                   cl::cat(ClangUnbundleArchiveCategory));
static cl::opt<std::string> FilesType("type", cl::Optional,
                                      cl::desc("Type of the device binaries.\n"
                                               "Current supported types are:\n"
                                               "  bc  - llvm-bc\n"
                                               "  o   - object\n"),
                                      cl::cat(ClangUnbundleArchiveCategory));
static cl::opt<std::string>
    OffloadArch("offload-arch", cl::Optional,
                cl::desc("Extract only libraries for a given device type\n"
                         "e.g. 'sm_60'\n"),
                cl::cat(ClangUnbundleArchiveCategory));
static cl::opt<bool>
    DryRun("dry-run",
           cl::desc("Extract archive and create new archives internally, "
                    "but do not write to files - for testing purposes.\n"),
           cl::init(false), cl::cat(ClangUnbundleArchiveCategory));

/// Magic string that marks the existence of offloading data.
#define OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"

// The name this program was invoked as.
static StringRef ToolName;

/// Obtain the offload kind and real machine triple out of the target
/// information specified by the user.
static void getOffloadKindAndTriple(StringRef Target, StringRef &OffloadKind,
                                    StringRef &Triple) {
  auto KindTriplePair = Target.split('-');
  OffloadKind = KindTriplePair.first;
  Triple = KindTriplePair.second;
}

static bool hasHostKind(StringRef Target) {
  StringRef OffloadKind;
  StringRef Triple;
  getOffloadKindAndTriple(Target, OffloadKind, Triple);
  return OffloadKind == "host";
}

static StringRef getTriple(StringRef Target) {
  StringRef OffloadKind;
  StringRef Triple;
  getOffloadKindAndTriple(Target, OffloadKind, Triple);
  return Triple;
}

/// Generic file handler interface.
class FileHandler {
public:
  FileHandler() {}

  virtual ~FileHandler() {}

  /// Update the file handler with information from the header of the bundled
  /// file
  virtual void ReadHeader(MemoryBuffer &Input) = 0;

  /// Read the marker of the next bundled to be read in the file. The triple of
  /// the target associated with that bundle is returned. An empty string is
  /// returned if there are no more bundles to be read.
  virtual StringRef ReadBundleStart(MemoryBuffer &Input) = 0;

  /// Read the marker that closes the current bundle.
  virtual void ReadBundleEnd(MemoryBuffer &Input) = 0;

  /// Read the current bundle and write the result into the stream \a OS.
  virtual void ReadBundle(raw_ostream &OS, MemoryBuffer &Input) = 0;
};

/// Handler for binary files. The bundled file will have the following format
/// (all integers are stored in little-endian format):
///
/// "OFFLOAD_BUNDLER_MAGIC_STR" (ASCII encoding of the string)
///
/// NumberOfOffloadBundles (8-byte integer)
///
/// OffsetOfBundle1 (8-byte integer)
/// SizeOfBundle1 (8-byte integer)
/// NumberOfBytesInTripleOfBundle1 (8-byte integer)
/// TripleOfBundle1 (byte length defined before)
///
/// ...
///
/// OffsetOfBundleN (8-byte integer)
/// SizeOfBundleN (8-byte integer)
/// NumberOfBytesInTripleOfBundleN (8-byte integer)
/// TripleOfBundleN (byte length defined before)
///
/// Bundle1
/// ...
/// BundleN

/// Read 8-byte integers from a buffer in little-endian format.
static uint64_t Read8byteIntegerFromBuffer(StringRef Buffer, size_t pos) {
  uint64_t Res = 0;
  const char *Data = Buffer.data();

  for (unsigned i = 0; i < 8; ++i) {
    Res <<= 8;
    uint64_t Char = (uint64_t)Data[pos + 7 - i];
    Res |= 0xffu & Char;
  }
  return Res;
}

class BinaryFileHandler final : public FileHandler {
  /// Information about the bundles extracted from the header.
  struct BundleInfo final {
    /// Size of the bundle.
    uint64_t Size = 0u;
    /// Offset at which the bundle starts in the bundled file.
    uint64_t Offset = 0u;

    BundleInfo() {}
    BundleInfo(uint64_t Size, uint64_t Offset) : Size(Size), Offset(Offset) {}
  };

  /// Map between a triple and the corresponding bundle information.
  StringMap<BundleInfo> BundlesInfo;

  /// Iterator for the bundle information that is being read.
  StringMap<BundleInfo>::iterator CurBundleInfo;

public:
  BinaryFileHandler() : FileHandler() {}

  ~BinaryFileHandler() final {}

  void ReadHeader(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Initialize the current bundle with the end of the container.
    CurBundleInfo = BundlesInfo.end();

    // Check if buffer is smaller than magic string.
    size_t ReadChars = sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;
    if (ReadChars > FC.size())
      return;

    // Check if no magic was found.
    StringRef Magic(FC.data(), sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
      return;

    // Read number of bundles.
    if (ReadChars + 8 > FC.size())
      return;

    uint64_t NumberOfBundles = Read8byteIntegerFromBuffer(FC, ReadChars);
    ReadChars += 8;

    // Read bundle offsets, sizes and triples.
    for (uint64_t i = 0; i < NumberOfBundles; ++i) {

      // Read offset.
      if (ReadChars + 8 > FC.size())
        return;

      uint64_t Offset = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read size.
      if (ReadChars + 8 > FC.size())
        return;

      uint64_t Size = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple size.
      if (ReadChars + 8 > FC.size())
        return;

      uint64_t TripleSize = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple.
      if (ReadChars + TripleSize > FC.size())
        return;

      StringRef Triple(&FC.data()[ReadChars], TripleSize);
      ReadChars += TripleSize;

      // Check if the offset and size make sense.
      if (!Size || !Offset || Offset + Size > FC.size())
        return;

      assert(BundlesInfo.find(Triple) == BundlesInfo.end() &&
             "Triple is duplicated??");
      BundlesInfo[Triple] = BundleInfo(Size, Offset);
    }
    // Set the iterator to where we will start to read.
    CurBundleInfo = BundlesInfo.begin();
  }

  StringRef ReadBundleStart(MemoryBuffer &Input) final {
    if (CurBundleInfo == BundlesInfo.end())
      return StringRef();

    return CurBundleInfo->first();
  }

  void ReadBundleEnd(MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    ++CurBundleInfo;
  }

  void ReadBundle(raw_ostream &OS, MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    StringRef FC = Input.getBuffer();
    OS.write(FC.data() + CurBundleInfo->second.Offset,
             CurBundleInfo->second.Size);
  }
};

class ObjectFileHandler final : public FileHandler {

  /// The object file we are currently dealing with.
  std::unique_ptr<ObjectFile> Obj;

  /// Return the input file contents.
  StringRef getInputFileContents() const { return Obj->getData(); }

  /// Return true if the provided section is an offload section and return the
  /// triple by reference.
  static bool IsOffloadSection(SectionRef CurSection,
                               StringRef &OffloadTriple) {

    Expected<StringRef> SectionName = CurSection.getName();

    if (!SectionName)
      return false;

    StringRef S = *SectionName;
    // If it does not start with the reserved suffix, just skip this section.
    if (!S.startswith(OFFLOAD_BUNDLER_MAGIC_STR))
      return false;

    // Return the triple that is right after the reserved prefix.
    OffloadTriple = S.substr(sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    return true;
  }

  /// Iterator of the current and next section.
  section_iterator CurrentSection;
  section_iterator NextSection;

public:
  ObjectFileHandler(std::unique_ptr<ObjectFile> ObjIn)
      : FileHandler(), Obj(std::move(ObjIn)),
        CurrentSection(Obj->section_begin()),
        NextSection(Obj->section_begin()) {}

  ~ObjectFileHandler() final {}

  void ReadHeader(MemoryBuffer &Input) final {}

  StringRef ReadBundleStart(MemoryBuffer &Input) final {
    while (NextSection != Obj->section_end()) {
      CurrentSection = NextSection;
      ++NextSection;

      StringRef OffloadTriple;
      // Check if the current section name starts with the reserved prefix. If
      // so, return the triple.
      if (IsOffloadSection(*CurrentSection, OffloadTriple))
        return OffloadTriple;
    }
    return StringRef();
  }

  void ReadBundleEnd(MemoryBuffer &Input) final {}

  void ReadBundle(raw_ostream &OS, MemoryBuffer &Input) final {
    // If the current section has size one, that means that the content we are
    // interested in is the file itself. Otherwise it is the content of the
    // section.
    //
    // TODO: Instead of copying the input file as is, deactivate the section
    // that is no longer needed.

    Expected<StringRef> ContentOrErr = CurrentSection->getContents();
    if (!ContentOrErr) {
      consumeError(ContentOrErr.takeError());
      return;
    }
    auto Content = ContentOrErr.get();
    if (Content.size() < 2)
      OS.write(Input.getBufferStart(), Input.getBufferSize());
    else
      OS.write(Content.data(), Content.size());
  }
};

/// Return an appropriate object file handler. We use the specific object
/// handler if we know how to deal with that format, otherwise we use a default
/// binary file handler.
static FileHandler *CreateObjectFileHandler(MemoryBuffer &FirstInput) {
  // Check if the input file format is one that we know how to deal with.
  Expected<std::unique_ptr<Binary>> BinaryOrErr = createBinary(FirstInput);

  // Failed to open the input as a known binary. Use the default binary handler.
  if (!BinaryOrErr) {
    // We don't really care about the error (we just consume it), if we could
    // not get a valid device binary object we use the default binary handler.
    consumeError(BinaryOrErr.takeError());
    return new BinaryFileHandler();
  }

  // We only support regular object files. If this is not an object file,
  // default to the binary handler. The handler will be owned by the client of
  // this function.
  std::unique_ptr<ObjectFile> Obj(
      dyn_cast<ObjectFile>(BinaryOrErr.get().release()));

  if (!Obj)
    return new BinaryFileHandler();

  return new ObjectFileHandler(std::move(Obj));
}

/// Return an appropriate handler given the input files and options.
static FileHandler *CreateFileHandler(MemoryBuffer &FirstInput) {
  return CreateObjectFileHandler(FirstInput);
}

// Show the error message and exit.
LLVM_ATTRIBUTE_NORETURN static void fail(Twine Error) {
  WithColor::error(errs(), ToolName) << Error << ".\n";
  exit(1);
}

static void failIfError(std::error_code EC, Twine Context = "") {
  if (!EC)
    return;

  std::string ContextStr = Context.str();
  if (ContextStr.empty())
    fail(EC.message());
  fail(Context + ": " + EC.message());
}

static void failIfError(Error E, Twine Context = "") {
  if (!E)
    return;

  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    std::string ContextStr = Context.str();
    if (ContextStr.empty())
      fail(EIB.message());
    fail(Context + ": " + EIB.message());
  });
}

std::vector<std::unique_ptr<MemoryBuffer>> ArchiveBuffers;
std::vector<std::unique_ptr<object::Archive>> Archives;

static object::Archive::Kind getDefaultArchiveKindForHost() {
  //return Triple(llvm::sys::getProcessTriple()).isOSDarwin()
  return false
             ? object::Archive::K_DARWIN
             : object::Archive::K_GNU;
}

static object::Archive &readArchive(std::unique_ptr<MemoryBuffer> Buf) {
  ArchiveBuffers.push_back(std::move(Buf));
  auto LibOrErr =
      object::Archive::create(ArchiveBuffers.back()->getMemBufferRef());
  failIfError(errorToErrorCode(LibOrErr.takeError()),
              "Could not parse library");
  Archives.push_back(std::move(*LibOrErr));
  return *Archives.back();
}

static StringRef getArchForDevice(StringRef Device) {
  if (Device.contains("gfx")) {
    return "amdgcn";
  } else {
    return "nvptx";
  }
}

static StringRef getDeviceFileExtension(StringRef Device) {
  if (FilesType.empty()) {
    if (Device.contains("gfx")) {
      return ".bc";
    } else {
      return ".cubin";
    }
  } else {
    return FilesType;
  }
}

static StringRef removeLibPrefix(StringRef FileName) {
  StringRef NoPrefixFileName;
  if (FileName.startswith("lib")) {
    NoPrefixFileName = FileName.slice(3, FileName.size());
  } else if (FileName.startswith("libbc")) {
    NoPrefixFileName = FileName.slice(5, FileName.size());
  } else {
    NoPrefixFileName = FileName;
  }
  return NoPrefixFileName;
}

static StringRef removeExtension(StringRef FileName) {
  StringRef NoExtFileName;
  if (FileName.contains(".")) {
    NoExtFileName = FileName.rsplit('.').first;
  } else {
    NoExtFileName = FileName;
  }
  return NoExtFileName;
}

static StringRef getLibPrefix(StringRef Extension) {
  return Extension == ".bc" ? "libbc-" : "lib";
}

static StringRef getDevice(StringRef Triple) {
  if (Triple.contains("-")) {
    auto Split = Triple.rsplit('-');
    return Split.second;
  } else {
    return Triple;
  }
}

// The getLibraryFileName function assumes that the bundle file name
// conforms to the SDL library convention:
// lib<libname>[-<arch-name>[-<device-type>]].o
static std::string getDeviceLibraryFileName(StringRef BundleFileName,
                                            StringRef Device) {
  StringRef LibNameExt = removeLibPrefix(sys::path::filename(BundleFileName));
  StringRef LibName = removeExtension(LibNameExt);
  StringRef Extension = getDeviceFileExtension(Device);
  StringRef LibPrefix = getLibPrefix(Extension);

  std::string Result;
  Result += LibPrefix;
  Result += LibName;
  Result += Extension;
  return Result;
}

static std::string getArchiveFileName(StringRef PathName, StringRef Device) {
  if (!OutputFile.empty()) {
    return OutputFile;
  }

  StringRef FileName = sys::path::filename(PathName);
  StringRef NoPrefixFileName = removeLibPrefix(FileName);
  StringRef NoExtFileName = removeExtension(NoPrefixFileName);

  std::string Result;
  Result += sys::path::parent_path(PathName);
  if (!Result.empty()) {
    Result += "/";
  }

  Result += getLibPrefix(getDeviceFileExtension(Device));
  Result += NoExtFileName;
  Result += "-";
  Result += getArchForDevice(Device);
  Result += "-";
  Result += Device;
  Result += ".a";

  return Result;
}

static bool checkDeviceOptions(StringRef Device) {
  bool Result(true);

  if (OffloadArch.empty() || OffloadArch != Device) {
    Result = false;
  }
  return Result;
  ;
}

// Unbundle the files. Return true if an error was found.
static bool UnbundleArchive() {
  std::map<std::string, std::vector<NewArchiveMember>> OutputArchives;

  StringRef IFName = InputFileName;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFileOrSTDIN(IFName, -1, false);
  failIfError(BufOrErr.getError(), "Can't open file " + IFName);

  if (!HostOutputFile.empty()) {
    std::error_code WriteEC;
    raw_fd_ostream HostFile(HostOutputFile, WriteEC);
    failIfError(WriteEC, "Unable create host archive file for write.");
    HostFile << BufOrErr.get()->getBuffer();
    HostFile.close();
  }

  auto &Archive = readArchive(std::move(BufOrErr.get()));

  Error Err = Error::success();
  auto ChildEnd = Archive.child_end();
  for (auto ChildIter = Archive.child_begin(Err); ChildIter != ChildEnd;
       ++ChildIter) {
    auto ChildNameOrErr = (*ChildIter).getName();
    failIfError(ChildNameOrErr.takeError(), "No Child Name");
    StringRef ChildName = sys::path::filename(ChildNameOrErr.get());

    auto ChildBufferRefOrErr = (*ChildIter).getMemoryBufferRef();
    failIfError(ChildBufferRefOrErr.takeError(), "No Child Mem Buf");
    auto ChildBuffer =
        MemoryBuffer::getMemBuffer(ChildBufferRefOrErr.get(), false);
    std::unique_ptr<FileHandler> FileHandler;
    FileHandler.reset(CreateFileHandler(*ChildBuffer));
    if (!FileHandler.get())
      return true;

    FileHandler.get()->ReadHeader(*ChildBuffer);

    StringRef CurKindTriple = FileHandler.get()->ReadBundleStart(*ChildBuffer);
    while (!CurKindTriple.empty()) {
      if (hasHostKind(CurKindTriple)) {
        // Do nothing, we don't extract host code yet
      } else if (checkDeviceOptions(getDevice(getTriple(CurKindTriple)))) {
        auto &TargetBundle = OutputArchives[OffloadArch];
        std::string BundleData;
        raw_string_ostream DataStream(BundleData);
        FileHandler.get()->ReadBundle(DataStream, *ChildBuffer);
        std::string *LibraryName =
            new std::string(getDeviceLibraryFileName(ChildName, OffloadArch));
        auto MemBuf =
            MemoryBuffer::getMemBufferCopy(DataStream.str(), *LibraryName);
        ArchiveBuffers.push_back(std::move(MemBuf));
        auto MemBufRef = MemoryBufferRef(*(ArchiveBuffers.back()));
        TargetBundle.push_back(NewArchiveMember(MemBufRef));
      }
      FileHandler.get()->ReadBundleEnd(*ChildBuffer);
      CurKindTriple = FileHandler.get()->ReadBundleStart(*ChildBuffer);
    }
  }

  if (!DryRun) {
    for (auto &Pair : OutputArchives) {
      auto &Device = Pair.first;
      auto &Members = Pair.second;
      std::string FileName = getArchiveFileName(IFName, Device);
      Error E =
          writeArchive(FileName, Members, true, getDefaultArchiveKindForHost(),
                       true, false, nullptr);
      failIfError(std::move(E), FileName);
    }
  }
  failIfError(std::move(Err));
  return OutputArchives.empty();
  ;
}

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-offload-bundler") << '\n';
}

void CheckFlags() {
  bool hasOutputFile = !OutputFile.empty();
  bool hasInputFile = !InputFileName.empty();
  bool hasOffloadArch = !OffloadArch.empty();

  if (!hasInputFile) {
    fail("error: no input file specified. use -input=<filename>");
  }

  if (!hasOutputFile && !DryRun) {
    fail("error: must specify output file use -output=<output file>.");
  }

  if (!hasOffloadArch) {
    fail("error: device not specified -offload-arch=<device>.");
  }
}

int main(int argc, const char **argv) {
  ToolName = argv[0];

  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangUnbundleArchiveCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to unbundle an archive of bundles contaning host and device "
      "code into individual archives for a given target.\n");
  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  CheckFlags();

  auto Result = UnbundleArchive();
  return Result;
}
