//===-- clang-offload-bundler/ClangOffloadBundler.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a clang-offload-bundler that bundles different
/// files that relate with the same source code but different targets into a
/// single one. Also the implements the opposite functionality, i.e. unbundle
/// files previous created by this tool.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadBundlerCategory("clang-offload-bundler options");

static cl::list<std::string>
    InputFileNames("inputs", cl::CommaSeparated, cl::OneOrMore,
                   cl::desc("[<input file>,...]"),
                   cl::cat(ClangOffloadBundlerCategory));
static cl::list<std::string>
    OutputFileNames("outputs", cl::CommaSeparated, cl::OneOrMore,
                    cl::desc("[<output file>,...]"),
                    cl::cat(ClangOffloadBundlerCategory));
static cl::list<std::string>
    TargetNames("targets", cl::CommaSeparated, cl::OneOrMore,
                cl::desc("[<offload kind>-<target triple>,...]"),
                cl::cat(ClangOffloadBundlerCategory));
static cl::opt<std::string>
    FilesType("type", cl::Required,
              cl::desc("Type of the files to be bundled/unbundled.\n"
                       "Current supported types are:\n"
                       "  i   - cpp-output\n"
                       "  ii  - c++-cpp-output\n"
                       "  cui - cuda/hip-output\n"
                       "  d   - dependency\n"
                       "  ll  - llvm\n"
                       "  bc  - llvm-bc\n"
                       "  s   - assembler\n"
                       "  o   - object\n"
                       "  gch - precompiled-header\n"
                       "  ast - clang AST file"),
              cl::cat(ClangOffloadBundlerCategory));
static cl::opt<bool>
    Unbundle("unbundle",
             cl::desc("Unbundle bundled file into several output files.\n"),
             cl::init(false), cl::cat(ClangOffloadBundlerCategory));

static cl::opt<bool> PrintExternalCommands(
    "###",
    cl::desc("Print any external commands that are to be executed "
             "instead of actually executing them - for testing purposes.\n"),
    cl::init(false), cl::cat(ClangOffloadBundlerCategory));

static cl::opt<unsigned>
    BundleAlignment("bundle-align",
                    cl::desc("Alignment of bundle for binary files"),
                    cl::init(1), cl::cat(ClangOffloadBundlerCategory));

/// Magic string that marks the existence of offloading data.
#define OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"

/// The index of the host input in the list of inputs.
static unsigned HostInputIndex = ~0u;

/// Path to the current binary.
static std::string BundlerExecutable;

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

/// Generic file handler interface.
class FileHandler {
public:
  FileHandler() {}

  virtual ~FileHandler() {}

  /// Update the file handler with information from the header of the bundled
  /// file.
  virtual Error ReadHeader(MemoryBuffer &Input) = 0;

  /// Read the marker of the next bundled to be read in the file. The bundle
  /// name is returned if there is one in the file, or `None` if there are no
  /// more bundles to be read.
  virtual Expected<Optional<StringRef>>
  ReadBundleStart(MemoryBuffer &Input) = 0;

  /// Read the marker that closes the current bundle.
  virtual Error ReadBundleEnd(MemoryBuffer &Input) = 0;

  /// Read the current bundle and write the result into the stream \a OS.
  virtual Error ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) = 0;

  /// Write the header of the bundled file to \a OS based on the information
  /// gathered from \a Inputs.
  virtual Error WriteHeader(raw_fd_ostream &OS,
                            ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) = 0;

  /// Write the marker that initiates a bundle for the triple \a TargetTriple to
  /// \a OS.
  virtual Error WriteBundleStart(raw_fd_ostream &OS,
                                 StringRef TargetTriple) = 0;

  /// Write the marker that closes a bundle for the triple \a TargetTriple to \a
  /// OS.
  virtual Error WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) = 0;

  /// Write the bundle from \a Input into \a OS.
  virtual Error WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) = 0;
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

/// Write 8-byte integers to a buffer in little-endian format.
static void Write8byteIntegerToBuffer(raw_fd_ostream &OS, uint64_t Val) {
  for (unsigned i = 0; i < 8; ++i) {
    char Char = (char)(Val & 0xffu);
    OS.write(&Char, 1);
    Val >>= 8;
  }
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
  StringMap<BundleInfo>::iterator NextBundleInfo;

  /// Current bundle target to be written.
  std::string CurWriteBundleTarget;

public:
  BinaryFileHandler() : FileHandler() {}

  ~BinaryFileHandler() final {}

  Error ReadHeader(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Initialize the current bundle with the end of the container.
    CurBundleInfo = BundlesInfo.end();

    // Check if buffer is smaller than magic string.
    size_t ReadChars = sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;
    if (ReadChars > FC.size())
      return Error::success();

    // Check if no magic was found.
    StringRef Magic(FC.data(), sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    if (!Magic.equals(OFFLOAD_BUNDLER_MAGIC_STR))
      return Error::success();

    // Read number of bundles.
    if (ReadChars + 8 > FC.size())
      return Error::success();

    uint64_t NumberOfBundles = Read8byteIntegerFromBuffer(FC, ReadChars);
    ReadChars += 8;

    // Read bundle offsets, sizes and triples.
    for (uint64_t i = 0; i < NumberOfBundles; ++i) {

      // Read offset.
      if (ReadChars + 8 > FC.size())
        return Error::success();

      uint64_t Offset = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read size.
      if (ReadChars + 8 > FC.size())
        return Error::success();

      uint64_t Size = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple size.
      if (ReadChars + 8 > FC.size())
        return Error::success();

      uint64_t TripleSize = Read8byteIntegerFromBuffer(FC, ReadChars);
      ReadChars += 8;

      // Read triple.
      if (ReadChars + TripleSize > FC.size())
        return Error::success();

      StringRef Triple(&FC.data()[ReadChars], TripleSize);
      ReadChars += TripleSize;

      // Check if the offset and size make sense.
      if (!Offset || Offset + Size > FC.size())
        return Error::success();

      assert(BundlesInfo.find(Triple) == BundlesInfo.end() &&
             "Triple is duplicated??");
      BundlesInfo[Triple] = BundleInfo(Size, Offset);
    }
    // Set the iterator to where we will start to read.
    CurBundleInfo = BundlesInfo.end();
    NextBundleInfo = BundlesInfo.begin();
    return Error::success();
  }

  Expected<Optional<StringRef>> ReadBundleStart(MemoryBuffer &Input) final {
    if (NextBundleInfo == BundlesInfo.end())
      return None;
    CurBundleInfo = NextBundleInfo++;
    return CurBundleInfo->first();
  }

  Error ReadBundleEnd(MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    return Error::success();
  }

  Error ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    StringRef FC = Input.getBuffer();
    OS.write(FC.data() + CurBundleInfo->second.Offset,
             CurBundleInfo->second.Size);
    return Error::success();
  }

  Error WriteHeader(raw_fd_ostream &OS,
                    ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) final {
    // Compute size of the header.
    uint64_t HeaderSize = 0;

    HeaderSize += sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1;
    HeaderSize += 8; // Number of Bundles

    for (auto &T : TargetNames) {
      HeaderSize += 3 * 8; // Bundle offset, Size of bundle and size of triple.
      HeaderSize += T.size(); // The triple.
    }

    // Write to the buffer the header.
    OS << OFFLOAD_BUNDLER_MAGIC_STR;

    Write8byteIntegerToBuffer(OS, TargetNames.size());

    unsigned Idx = 0;
    for (auto &T : TargetNames) {
      MemoryBuffer &MB = *Inputs[Idx++];
      HeaderSize = alignTo(HeaderSize, BundleAlignment);
      // Bundle offset.
      Write8byteIntegerToBuffer(OS, HeaderSize);
      // Size of the bundle (adds to the next bundle's offset)
      Write8byteIntegerToBuffer(OS, MB.getBufferSize());
      BundlesInfo[T] = BundleInfo(MB.getBufferSize(), HeaderSize);
      HeaderSize += MB.getBufferSize();
      // Size of the triple
      Write8byteIntegerToBuffer(OS, T.size());
      // Triple
      OS << T;
    }
    return Error::success();
  }

  Error WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) final {
    CurWriteBundleTarget = TargetTriple.str();
    return Error::success();
  }

  Error WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) final {
    return Error::success();
  }

  Error WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    auto BI = BundlesInfo[CurWriteBundleTarget];
    OS.seek(BI.Offset);
    OS.write(Input.getBufferStart(), Input.getBufferSize());
    return Error::success();
  }
};

namespace {

// This class implements a list of temporary files that are removed upon
// object destruction.
class TempFileHandlerRAII {
public:
  ~TempFileHandlerRAII() {
    for (const auto &File : Files)
      sys::fs::remove(File);
  }

  // Creates temporary file with given contents.
  Expected<StringRef> Create(Optional<ArrayRef<char>> Contents) {
    SmallString<128u> File;
    if (std::error_code EC =
            sys::fs::createTemporaryFile("clang-offload-bundler", "tmp", File))
      return createFileError(File, EC);
    Files.push_front(File);

    if (Contents) {
      std::error_code EC;
      raw_fd_ostream OS(File, EC);
      if (EC)
        return createFileError(File, EC);
      OS.write(Contents->data(), Contents->size());
    }
    return Files.front();
  }

private:
  std::forward_list<SmallString<128u>> Files;
};

} // end anonymous namespace

/// Handler for object files. The bundles are organized by sections with a
/// designated name.
///
/// To unbundle, we just copy the contents of the designated section.
class ObjectFileHandler final : public FileHandler {

  /// The object file we are currently dealing with.
  std::unique_ptr<ObjectFile> Obj;

  /// Return the input file contents.
  StringRef getInputFileContents() const { return Obj->getData(); }

  /// Return bundle name (<kind>-<triple>) if the provided section is an offload
  /// section.
  static Expected<Optional<StringRef>> IsOffloadSection(SectionRef CurSection) {
    Expected<StringRef> NameOrErr = CurSection.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();

    // If it does not start with the reserved suffix, just skip this section.
    if (!NameOrErr->startswith(OFFLOAD_BUNDLER_MAGIC_STR))
      return None;

    // Return the triple that is right after the reserved prefix.
    return NameOrErr->substr(sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  }

  /// Total number of inputs.
  unsigned NumberOfInputs = 0;

  /// Total number of processed inputs, i.e, inputs that were already
  /// read from the buffers.
  unsigned NumberOfProcessedInputs = 0;

  /// Iterator of the current and next section.
  section_iterator CurrentSection;
  section_iterator NextSection;

public:
  ObjectFileHandler(std::unique_ptr<ObjectFile> ObjIn)
      : FileHandler(), Obj(std::move(ObjIn)),
        CurrentSection(Obj->section_begin()),
        NextSection(Obj->section_begin()) {}

  ~ObjectFileHandler() final {}

  Error ReadHeader(MemoryBuffer &Input) final { return Error::success(); }

  Expected<Optional<StringRef>> ReadBundleStart(MemoryBuffer &Input) final {
    while (NextSection != Obj->section_end()) {
      CurrentSection = NextSection;
      ++NextSection;

      // Check if the current section name starts with the reserved prefix. If
      // so, return the triple.
      Expected<Optional<StringRef>> TripleOrErr =
          IsOffloadSection(*CurrentSection);
      if (!TripleOrErr)
        return TripleOrErr.takeError();
      if (*TripleOrErr)
        return **TripleOrErr;
    }
    return None;
  }

  Error ReadBundleEnd(MemoryBuffer &Input) final { return Error::success(); }

  Error ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    Expected<StringRef> ContentOrErr = CurrentSection->getContents();
    if (!ContentOrErr)
      return ContentOrErr.takeError();
    StringRef Content = *ContentOrErr;

    // Copy fat object contents to the output when extracting host bundle.
    if (Content.size() == 1u && Content.front() == 0)
      Content = StringRef(Input.getBufferStart(), Input.getBufferSize());

    OS.write(Content.data(), Content.size());
    return Error::success();
  }

  Error WriteHeader(raw_fd_ostream &OS,
                    ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) final {
    assert(HostInputIndex != ~0u && "Host input index not defined.");

    // Record number of inputs.
    NumberOfInputs = Inputs.size();
    return Error::success();
  }

  Error WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) final {
    ++NumberOfProcessedInputs;
    return Error::success();
  }

  Error WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) final {
    assert(NumberOfProcessedInputs <= NumberOfInputs &&
           "Processing more inputs that actually exist!");
    assert(HostInputIndex != ~0u && "Host input index not defined.");

    // If this is not the last output, we don't have to do anything.
    if (NumberOfProcessedInputs != NumberOfInputs)
      return Error::success();

    // We will use llvm-objcopy to add target objects sections to the output
    // fat object. These sections should have 'exclude' flag set which tells
    // link editor to remove them from linker inputs when linking executable or
    // shared library. llvm-objcopy currently does not support adding new
    // section and changing flags for the added section in one invocation, and
    // because of that we have to run it two times. First run adds sections and
    // the second changes flags.
    // TODO: change it to one run once llvm-objcopy starts supporting that.

    // Find llvm-objcopy in order to create the bundle binary.
    ErrorOr<std::string> Objcopy = sys::findProgramByName(
        "llvm-objcopy", sys::path::parent_path(BundlerExecutable));
    if (!Objcopy)
      Objcopy = sys::findProgramByName("llvm-objcopy");
    if (!Objcopy)
      return createStringError(Objcopy.getError(),
                               "unable to find 'llvm-objcopy' in path");

    // We write to the output file directly. So, we close it and use the name
    // to pass down to llvm-objcopy.
    OS.close();

    // Temporary files that need to be removed.
    TempFileHandlerRAII TempFiles;

    // Create an intermediate temporary file to save object after the first
    // llvm-objcopy run.
    Expected<StringRef> IntermediateObjOrErr = TempFiles.Create(None);
    if (!IntermediateObjOrErr)
      return IntermediateObjOrErr.takeError();
    StringRef IntermediateObj = *IntermediateObjOrErr;

    // Compose llvm-objcopy command line for add target objects' sections.
    BumpPtrAllocator Alloc;
    StringSaver SS{Alloc};
    SmallVector<StringRef, 8u> ObjcopyArgs{"llvm-objcopy"};
    for (unsigned I = 0; I < NumberOfInputs; ++I) {
      StringRef InputFile = InputFileNames[I];
      if (I == HostInputIndex) {
        // Special handling for the host bundle. We do not need to add a
        // standard bundle for the host object since we are going to use fat
        // object as a host object. Therefore use dummy contents (one zero byte)
        // when creating section for the host bundle.
        Expected<StringRef> TempFileOrErr = TempFiles.Create(ArrayRef<char>(0));
        if (!TempFileOrErr)
          return TempFileOrErr.takeError();
        InputFile = *TempFileOrErr;
      }

      ObjcopyArgs.push_back(SS.save(Twine("--add-section=") +
                                    OFFLOAD_BUNDLER_MAGIC_STR + TargetNames[I] +
                                    "=" + InputFile));
    }
    ObjcopyArgs.push_back(InputFileNames[HostInputIndex]);
    ObjcopyArgs.push_back(IntermediateObj);

    if (Error Err = executeObjcopy(*Objcopy, ObjcopyArgs))
      return Err;

    // And run llvm-objcopy for the second time to update section flags.
    ObjcopyArgs.resize(1);
    for (unsigned I = 0; I < NumberOfInputs; ++I)
      ObjcopyArgs.push_back(SS.save(Twine("--set-section-flags=") +
                                    OFFLOAD_BUNDLER_MAGIC_STR + TargetNames[I] +
                                    "=readonly,exclude"));
    ObjcopyArgs.push_back(IntermediateObj);
    ObjcopyArgs.push_back(OutputFileNames.front());

    if (Error Err = executeObjcopy(*Objcopy, ObjcopyArgs))
      return Err;

    return Error::success();
  }

  Error WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    return Error::success();
  }

private:
  static Error executeObjcopy(StringRef Objcopy, ArrayRef<StringRef> Args) {
    // If the user asked for the commands to be printed out, we do that
    // instead of executing it.
    if (PrintExternalCommands) {
      errs() << "\"" << Objcopy << "\"";
      for (StringRef Arg : drop_begin(Args, 1))
        errs() << " \"" << Arg << "\"";
      errs() << "\n";
    } else {
      if (sys::ExecuteAndWait(Objcopy, Args))
        return createStringError(inconvertibleErrorCode(),
                                 "'llvm-objcopy' tool failed");
    }
    return Error::success();
  }
};

/// Handler for text files. The bundled file will have the following format.
///
/// "Comment OFFLOAD_BUNDLER_MAGIC_STR__START__ triple"
/// Bundle 1
/// "Comment OFFLOAD_BUNDLER_MAGIC_STR__END__ triple"
/// ...
/// "Comment OFFLOAD_BUNDLER_MAGIC_STR__START__ triple"
/// Bundle N
/// "Comment OFFLOAD_BUNDLER_MAGIC_STR__END__ triple"
class TextFileHandler final : public FileHandler {
  /// String that begins a line comment.
  StringRef Comment;

  /// String that initiates a bundle.
  std::string BundleStartString;

  /// String that closes a bundle.
  std::string BundleEndString;

  /// Number of chars read from input.
  size_t ReadChars = 0u;

protected:
  Error ReadHeader(MemoryBuffer &Input) final { return Error::success(); }

  Expected<Optional<StringRef>> ReadBundleStart(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Find start of the bundle.
    ReadChars = FC.find(BundleStartString, ReadChars);
    if (ReadChars == FC.npos)
      return None;

    // Get position of the triple.
    size_t TripleStart = ReadChars = ReadChars + BundleStartString.size();

    // Get position that closes the triple.
    size_t TripleEnd = ReadChars = FC.find("\n", ReadChars);
    if (TripleEnd == FC.npos)
      return None;

    // Next time we read after the new line.
    ++ReadChars;

    return StringRef(&FC.data()[TripleStart], TripleEnd - TripleStart);
  }

  Error ReadBundleEnd(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Read up to the next new line.
    assert(FC[ReadChars] == '\n' && "The bundle should end with a new line.");

    size_t TripleEnd = ReadChars = FC.find("\n", ReadChars + 1);
    if (TripleEnd != FC.npos)
      // Next time we read after the new line.
      ++ReadChars;

    return Error::success();
  }

  Error ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();
    size_t BundleStart = ReadChars;

    // Find end of the bundle.
    size_t BundleEnd = ReadChars = FC.find(BundleEndString, ReadChars);

    StringRef Bundle(&FC.data()[BundleStart], BundleEnd - BundleStart);
    OS << Bundle;

    return Error::success();
  }

  Error WriteHeader(raw_fd_ostream &OS,
                    ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) final {
    return Error::success();
  }

  Error WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) final {
    OS << BundleStartString << TargetTriple << "\n";
    return Error::success();
  }

  Error WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) final {
    OS << BundleEndString << TargetTriple << "\n";
    return Error::success();
  }

  Error WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    OS << Input.getBuffer();
    return Error::success();
  }

public:
  TextFileHandler(StringRef Comment)
      : FileHandler(), Comment(Comment), ReadChars(0) {
    BundleStartString =
        "\n" + Comment.str() + " " OFFLOAD_BUNDLER_MAGIC_STR "__START__ ";
    BundleEndString =
        "\n" + Comment.str() + " " OFFLOAD_BUNDLER_MAGIC_STR "__END__ ";
  }
};

/// Return an appropriate object file handler. We use the specific object
/// handler if we know how to deal with that format, otherwise we use a default
/// binary file handler.
static std::unique_ptr<FileHandler>
CreateObjectFileHandler(MemoryBuffer &FirstInput) {
  // Check if the input file format is one that we know how to deal with.
  Expected<std::unique_ptr<Binary>> BinaryOrErr = createBinary(FirstInput);

  // We only support regular object files. If failed to open the input as a
  // known binary or this is not an object file use the default binary handler.
  if (errorToBool(BinaryOrErr.takeError()) || !isa<ObjectFile>(*BinaryOrErr))
    return std::make_unique<BinaryFileHandler>();

  // Otherwise create an object file handler. The handler will be owned by the
  // client of this function.
  return std::make_unique<ObjectFileHandler>(
      std::unique_ptr<ObjectFile>(cast<ObjectFile>(BinaryOrErr->release())));
}

/// Return an appropriate handler given the input files and options.
static Expected<std::unique_ptr<FileHandler>>
CreateFileHandler(MemoryBuffer &FirstInput) {
  if (FilesType == "i")
    return std::make_unique<TextFileHandler>(/*Comment=*/"//");
  if (FilesType == "ii")
    return std::make_unique<TextFileHandler>(/*Comment=*/"//");
  if (FilesType == "cui")
    return std::make_unique<TextFileHandler>(/*Comment=*/"//");
  // TODO: `.d` should be eventually removed once `-M` and its variants are
  // handled properly in offload compilation.
  if (FilesType == "d")
    return std::make_unique<TextFileHandler>(/*Comment=*/"#");
  if (FilesType == "ll")
    return std::make_unique<TextFileHandler>(/*Comment=*/";");
  if (FilesType == "bc")
    return std::make_unique<BinaryFileHandler>();
  if (FilesType == "s")
    return std::make_unique<TextFileHandler>(/*Comment=*/"#");
  if (FilesType == "o")
    return CreateObjectFileHandler(FirstInput);
  if (FilesType == "gch")
    return std::make_unique<BinaryFileHandler>();
  if (FilesType == "ast")
    return std::make_unique<BinaryFileHandler>();

  return createStringError(errc::invalid_argument,
                           "'" + FilesType + "': invalid file type specified");
}

/// Bundle the files. Return true if an error was found.
static Error BundleFiles() {
  std::error_code EC;

  // Create output file.
  raw_fd_ostream OutputFile(OutputFileNames.front(), EC, sys::fs::OF_None);
  if (EC)
    return createFileError(OutputFileNames.front(), EC);

  // Open input files.
  SmallVector<std::unique_ptr<MemoryBuffer>, 8u> InputBuffers;
  InputBuffers.reserve(InputFileNames.size());
  for (auto &I : InputFileNames) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
        MemoryBuffer::getFileOrSTDIN(I);
    if (std::error_code EC = CodeOrErr.getError())
      return createFileError(I, EC);
    InputBuffers.emplace_back(std::move(*CodeOrErr));
  }

  // Get the file handler. We use the host buffer as reference.
  assert(HostInputIndex != ~0u && "Host input index undefined??");
  Expected<std::unique_ptr<FileHandler>> FileHandlerOrErr =
      CreateFileHandler(*InputBuffers[HostInputIndex]);
  if (!FileHandlerOrErr)
    return FileHandlerOrErr.takeError();

  std::unique_ptr<FileHandler> &FH = *FileHandlerOrErr;
  assert(FH);

  // Write header.
  if (Error Err = FH->WriteHeader(OutputFile, InputBuffers))
    return Err;

  // Write all bundles along with the start/end markers. If an error was found
  // writing the end of the bundle component, abort the bundle writing.
  auto Input = InputBuffers.begin();
  for (auto &Triple : TargetNames) {
    if (Error Err = FH->WriteBundleStart(OutputFile, Triple))
      return Err;
    if (Error Err = FH->WriteBundle(OutputFile, **Input))
      return Err;
    if (Error Err = FH->WriteBundleEnd(OutputFile, Triple))
      return Err;
    ++Input;
  }
  return Error::success();
}

// Unbundle the files. Return true if an error was found.
static Error UnbundleFiles() {
  // Open Input file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFileNames.front());
  if (std::error_code EC = CodeOrErr.getError())
    return createFileError(InputFileNames.front(), EC);

  MemoryBuffer &Input = **CodeOrErr;

  // Select the right files handler.
  Expected<std::unique_ptr<FileHandler>> FileHandlerOrErr =
      CreateFileHandler(Input);
  if (!FileHandlerOrErr)
    return FileHandlerOrErr.takeError();

  std::unique_ptr<FileHandler> &FH = *FileHandlerOrErr;
  assert(FH);

  // Read the header of the bundled file.
  if (Error Err = FH->ReadHeader(Input))
    return Err;

  // Create a work list that consist of the map triple/output file.
  StringMap<StringRef> Worklist;
  auto Output = OutputFileNames.begin();
  for (auto &Triple : TargetNames) {
    Worklist[Triple] = *Output;
    ++Output;
  }

  // Read all the bundles that are in the work list. If we find no bundles we
  // assume the file is meant for the host target.
  bool FoundHostBundle = false;
  while (!Worklist.empty()) {
    Expected<Optional<StringRef>> CurTripleOrErr = FH->ReadBundleStart(Input);
    if (!CurTripleOrErr)
      return CurTripleOrErr.takeError();

    // We don't have more bundles.
    if (!*CurTripleOrErr)
      break;

    StringRef CurTriple = **CurTripleOrErr;
    assert(!CurTriple.empty());

    auto Output = Worklist.find(CurTriple);
    // The file may have more bundles for other targets, that we don't care
    // about. Therefore, move on to the next triple
    if (Output == Worklist.end())
      continue;

    // Check if the output file can be opened and copy the bundle to it.
    std::error_code EC;
    raw_fd_ostream OutputFile(Output->second, EC, sys::fs::OF_None);
    if (EC)
      return createFileError(Output->second, EC);
    if (Error Err = FH->ReadBundle(OutputFile, Input))
      return Err;
    if (Error Err = FH->ReadBundleEnd(Input))
      return Err;
    Worklist.erase(Output);

    // Record if we found the host bundle.
    if (hasHostKind(CurTriple))
      FoundHostBundle = true;
  }

  // If no bundles were found, assume the input file is the host bundle and
  // create empty files for the remaining targets.
  if (Worklist.size() == TargetNames.size()) {
    for (auto &E : Worklist) {
      std::error_code EC;
      raw_fd_ostream OutputFile(E.second, EC, sys::fs::OF_None);
      if (EC)
        return createFileError(E.second, EC);

      // If this entry has a host kind, copy the input file to the output file.
      if (hasHostKind(E.first()))
        OutputFile.write(Input.getBufferStart(), Input.getBufferSize());
    }
    return Error::success();
  }

  // If we found elements, we emit an error if none of those were for the host
  // in case host bundle name was provided in command line.
  if (!FoundHostBundle && HostInputIndex != ~0u)
    return createStringError(inconvertibleErrorCode(),
                             "Can't find bundle for the host target");

  // If we still have any elements in the worklist, create empty files for them.
  for (auto &E : Worklist) {
    std::error_code EC;
    raw_fd_ostream OutputFile(E.second, EC, sys::fs::OF_None);
    if (EC)
      return createFileError(E.second, EC);
  }

  return Error::success();
}

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-offload-bundler") << '\n';
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadBundlerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to bundle several input files of the specified type <type> \n"
      "referring to the same source file but different targets into a single \n"
      "one. The resulting file can also be unbundled into different files by \n"
      "this tool if -unbundle is provided.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
  };

  bool Error = false;
  if (Unbundle) {
    if (InputFileNames.size() != 1) {
      Error = true;
      reportError(createStringError(
          errc::invalid_argument,
          "only one input file supported in unbundling mode"));
    }
    if (OutputFileNames.size() != TargetNames.size()) {
      Error = true;
      reportError(createStringError(errc::invalid_argument,
                                    "number of output files and targets should "
                                    "match in unbundling mode"));
    }
  } else {
    if (OutputFileNames.size() != 1) {
      Error = true;
      reportError(createStringError(
          errc::invalid_argument,
          "only one output file supported in bundling mode"));
    }
    if (InputFileNames.size() != TargetNames.size()) {
      Error = true;
      reportError(createStringError(
          errc::invalid_argument,
          "number of input files and targets should match in bundling mode"));
    }
  }

  // Verify that the offload kinds and triples are known. We also check that we
  // have exactly one host target.
  unsigned Index = 0u;
  unsigned HostTargetNum = 0u;
  for (StringRef Target : TargetNames) {
    StringRef Kind;
    StringRef Triple;
    getOffloadKindAndTriple(Target, Kind, Triple);

    bool KindIsValid = !Kind.empty();
    KindIsValid = KindIsValid && StringSwitch<bool>(Kind)
                                     .Case("host", true)
                                     .Case("openmp", true)
                                     .Case("hip", true)
                                     .Default(false);

    bool TripleIsValid = !Triple.empty();
    llvm::Triple T(Triple);
    TripleIsValid &= T.getArch() != Triple::UnknownArch;

    if (!KindIsValid || !TripleIsValid) {
      Error = true;

      SmallVector<char, 128u> Buf;
      raw_svector_ostream Msg(Buf);
      Msg << "invalid target '" << Target << "'";
      if (!KindIsValid)
        Msg << ", unknown offloading kind '" << Kind << "'";
      if (!TripleIsValid)
        Msg << ", unknown target triple '" << Triple << "'";
      reportError(createStringError(errc::invalid_argument, Msg.str()));
    }

    if (KindIsValid && Kind == "host") {
      ++HostTargetNum;
      // Save the index of the input that refers to the host.
      HostInputIndex = Index;
    }

    ++Index;
  }

  // Host triple is not really needed for unbundling operation, so do not
  // treat missing host triple as error if we do unbundling.
  if ((Unbundle && HostTargetNum > 1) || (!Unbundle && HostTargetNum != 1)) {
    Error = true;
    reportError(createStringError(errc::invalid_argument,
                                  "expecting exactly one host target but got " +
                                      Twine(HostTargetNum)));
  }

  if (Error)
    return 1;

  // Save the current executable directory as it will be useful to find other
  // tools.
  BundlerExecutable = argv[0];
  if (!llvm::sys::fs::exists(BundlerExecutable))
    BundlerExecutable = sys::fs::getMainExecutable(argv[0], &BundlerExecutable);

  if (llvm::Error Err = Unbundle ? UnbundleFiles() : BundleFiles()) {
    reportError(std::move(Err));
    return 1;
  }
  return 0;
}
