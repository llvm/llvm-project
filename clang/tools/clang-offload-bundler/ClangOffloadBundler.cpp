//===-- clang-offload-bundler/ClangOffloadBundler.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Signals.h"
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

static cl::opt<bool> DumpTemporaryFiles(
    "dump-temporary-files",
    cl::desc("Dumps any temporary files created - for testing purposes.\n"),
    cl::init(false), cl::cat(ClangOffloadBundlerCategory));

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
static StringRef getTriple(StringRef Target) {
  StringRef OffloadKind;
  StringRef Triple;
  getOffloadKindAndTriple(Target, OffloadKind, Triple);
  return Triple;
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
  /// file
  virtual void ReadHeader(MemoryBuffer &Input) = 0;

  /// Read the marker of the next bundled to be read in the file. The triple of
  /// the target associated with that bundle is returned. An empty string is
  /// returned if there are no more bundles to be read.
  virtual StringRef ReadBundleStart(MemoryBuffer &Input) = 0;

  /// Read the marker that closes the current bundle.
  virtual void ReadBundleEnd(MemoryBuffer &Input) = 0;

  /// Read the current bundle and write the result into the stream \a OS.
  virtual void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) = 0;

  /// Write the header of the bundled file to \a OS based on the information
  /// gathered from \a Inputs.
  virtual void WriteHeader(raw_fd_ostream &OS,
                           ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) = 0;

  /// Write the marker that initiates a bundle for the triple \a TargetTriple to
  /// \a OS.
  virtual void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) = 0;

  /// Write the marker that closes a bundle for the triple \a TargetTriple to \a
  /// OS. Return true if any error was found.

  virtual bool WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) = 0;

  /// Write the bundle from \a Input into \a OS.
  virtual void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) = 0;
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

  void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    assert(CurBundleInfo != BundlesInfo.end() && "Invalid reader info!");
    StringRef FC = Input.getBuffer();
    OS.write(FC.data() + CurBundleInfo->second.Offset,
             CurBundleInfo->second.Size);
  }

  void WriteHeader(raw_fd_ostream &OS,
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
      MemoryBuffer &MB = *Inputs[Idx++].get();
      // Bundle offset.
      Write8byteIntegerToBuffer(OS, HeaderSize);
      // Size of the bundle (adds to the next bundle's offset)
      Write8byteIntegerToBuffer(OS, MB.getBufferSize());
      HeaderSize += MB.getBufferSize();
      // Size of the triple
      Write8byteIntegerToBuffer(OS, T.size());
      // Triple
      OS << T;
    }
  }

  void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) final {}

  bool WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) final {
    return false;
  }

  void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    OS.write(Input.getBufferStart(), Input.getBufferSize());
  }
};

/// Handler for object files. The bundles are organized by sections with a
/// designated name.
///
/// In order to bundle we create an IR file with the content of each section and
/// use incremental linking to produce the resulting object. We also add section
/// with a single byte to state the name of the component the main object file
/// (the one we are bundling into) refers to.
///
/// To unbundle, we use just copy the contents of the designated section. If the
/// requested bundle refer to the main object file, we just copy it with no
/// changes.
class ObjectFileHandler final : public FileHandler {

  /// The object file we are currently dealing with.
  std::unique_ptr<ObjectFile> Obj;

  /// Return the input file contents.
  StringRef getInputFileContents() const { return Obj->getData(); }

  /// Return true if the provided section is an offload section and return the
  /// triple by reference.
  static bool IsOffloadSection(SectionRef CurSection,
                               StringRef &OffloadTriple) {
    StringRef SectionName;
    CurSection.getName(SectionName);

    if (SectionName.empty())
      return false;

    // If it does not start with the reserved suffix, just skip this section.
    if (!SectionName.startswith(OFFLOAD_BUNDLER_MAGIC_STR))
      return false;

    // Return the triple that is right after the reserved prefix.
    OffloadTriple = SectionName.substr(sizeof(OFFLOAD_BUNDLER_MAGIC_STR) - 1);
    return true;
  }

  /// Total number of inputs.
  unsigned NumberOfInputs = 0;

  /// Total number of processed inputs, i.e, inputs that were already
  /// read from the buffers.
  unsigned NumberOfProcessedInputs = 0;

  /// LLVM context used to create the auxiliary modules.
  LLVMContext VMContext;

  /// LLVM module used to create an object with all the bundle
  /// components.
  std::unique_ptr<Module> AuxModule;

  /// The current triple we are working with.
  StringRef CurrentTriple;

  /// The name of the main input file.
  StringRef MainInputFileName;

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

  void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    // If the current section has size one, that means that the content we are
    // interested in is the file itself. Otherwise it is the content of the
    // section.
    //
    // TODO: Instead of copying the input file as is, deactivate the section
    // that is no longer needed.

    StringRef Content;
    CurrentSection->getContents(Content);

    if (Content.size() < 2)
      OS.write(Input.getBufferStart(), Input.getBufferSize());
    else
      OS.write(Content.data(), Content.size());
  }

  void WriteHeader(raw_fd_ostream &OS,
                   ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) final {
    assert(HostInputIndex != ~0u && "Host input index not defined.");

    // Record number of inputs.
    NumberOfInputs = Inputs.size();

    // Create an LLVM module to have the content we need to bundle.
    auto *M = new Module("clang-offload-bundle", VMContext);
    M->setTargetTriple(getTriple(TargetNames[HostInputIndex]));
    AuxModule.reset(M);
  }

  void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) final {
    ++NumberOfProcessedInputs;

    // Record the triple we are using, that will be used to name the section we
    // will create.
    CurrentTriple = TargetTriple;
  }

  bool WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) final {
    assert(NumberOfProcessedInputs <= NumberOfInputs &&
           "Processing more inputs that actually exist!");
    assert(HostInputIndex != ~0u && "Host input index not defined.");

    // If this is not the last output, we don't have to do anything.
    if (NumberOfProcessedInputs != NumberOfInputs)
      return false;

    // Create the bitcode file name to write the resulting code to. Keep it if
    // save-temps is active.
    SmallString<128> BitcodeFileName;
    if (sys::fs::createTemporaryFile("clang-offload-bundler", "bc",
                                     BitcodeFileName)) {
      errs() << "error: unable to create temporary file.\n";
      return true;
    }

    // Dump the contents of the temporary file if that was requested.
    if (DumpTemporaryFiles) {
      errs() << ";\n; Object file bundler IR file.\n;\n";
      AuxModule.get()->print(errs(), nullptr,
                             /*ShouldPreserveUseListOrder=*/false,
                             /*IsForDebug=*/true);
      errs() << '\n';
    }

    // Find clang in order to create the bundle binary.
    StringRef Dir = sys::path::parent_path(BundlerExecutable);

    auto ClangBinary = sys::findProgramByName("clang", Dir);
    if (ClangBinary.getError()) {
      // Remove bitcode file.
      sys::fs::remove(BitcodeFileName);

      errs() << "error: unable to find 'clang' in path.\n";
      return true;
    }

    // Do the incremental linking. We write to the output file directly. So, we
    // close it and use the name to pass down to clang.
    OS.close();
    SmallString<128> TargetName = getTriple(TargetNames[HostInputIndex]);
    std::vector<StringRef> ClangArgs = {"clang",
                                        "-r",
                                        "-target",
                                        TargetName.c_str(),
                                        "-o",
                                        OutputFileNames.front().c_str(),
                                        InputFileNames[HostInputIndex].c_str(),
                                        BitcodeFileName.c_str(),
                                        "-nostdlib"};

    // If the user asked for the commands to be printed out, we do that instead
    // of executing it.
    if (PrintExternalCommands) {
      errs() << "\"" << ClangBinary.get() << "\"";
      for (StringRef Arg : ClangArgs)
        errs() << " \"" << Arg << "\"";
      errs() << "\n";
    } else {
      // Write the bitcode contents to the temporary file.
      {
        std::error_code EC;
        raw_fd_ostream BitcodeFile(BitcodeFileName, EC, sys::fs::F_None);
        if (EC) {
          errs() << "error: unable to open temporary file.\n";
          return true;
        }
        WriteBitcodeToFile(*AuxModule, BitcodeFile);
      }

      bool Failed = sys::ExecuteAndWait(ClangBinary.get(), ClangArgs);

      // Remove bitcode file.
      sys::fs::remove(BitcodeFileName);

      if (Failed) {
        errs() << "error: incremental linking by external tool failed.\n";
        return true;
      }
    }

    return false;
  }

  void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    Module *M = AuxModule.get();

    // Create the new section name, it will consist of the reserved prefix
    // concatenated with the triple.
    std::string SectionName = OFFLOAD_BUNDLER_MAGIC_STR;
    SectionName += CurrentTriple;

    // Create the constant with the content of the section. For the input we are
    // bundling into (the host input), this is just a place-holder, so a single
    // byte is sufficient.
    assert(HostInputIndex != ~0u && "Host input index undefined??");
    Constant *Content;
    if (NumberOfProcessedInputs == HostInputIndex + 1) {
      uint8_t Byte[] = {0};
      Content = ConstantDataArray::get(VMContext, Byte);
    } else
      Content = ConstantDataArray::get(
          VMContext, ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(
                                           Input.getBufferStart()),
                                       Input.getBufferSize()));

    // Create the global in the desired section. We don't want these globals in
    // the symbol table, so we mark them private.
    auto *GV = new GlobalVariable(*M, Content->getType(), /*IsConstant=*/true,
                                  GlobalVariable::PrivateLinkage, Content);
    GV->setSection(SectionName);
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
  void ReadHeader(MemoryBuffer &Input) final {}

  StringRef ReadBundleStart(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Find start of the bundle.
    ReadChars = FC.find(BundleStartString, ReadChars);
    if (ReadChars == FC.npos)
      return StringRef();

    // Get position of the triple.
    size_t TripleStart = ReadChars = ReadChars + BundleStartString.size();

    // Get position that closes the triple.
    size_t TripleEnd = ReadChars = FC.find("\n", ReadChars);
    if (TripleEnd == FC.npos)
      return StringRef();

    // Next time we read after the new line.
    ++ReadChars;

    return StringRef(&FC.data()[TripleStart], TripleEnd - TripleStart);
  }

  void ReadBundleEnd(MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();

    // Read up to the next new line.
    assert(FC[ReadChars] == '\n' && "The bundle should end with a new line.");

    size_t TripleEnd = ReadChars = FC.find("\n", ReadChars + 1);
    if (TripleEnd == FC.npos)
      return;

    // Next time we read after the new line.
    ++ReadChars;
  }

  void ReadBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    StringRef FC = Input.getBuffer();
    size_t BundleStart = ReadChars;

    // Find end of the bundle.
    size_t BundleEnd = ReadChars = FC.find(BundleEndString, ReadChars);

    StringRef Bundle(&FC.data()[BundleStart], BundleEnd - BundleStart);
    OS << Bundle;
  }

  void WriteHeader(raw_fd_ostream &OS,
                   ArrayRef<std::unique_ptr<MemoryBuffer>> Inputs) final {}

  void WriteBundleStart(raw_fd_ostream &OS, StringRef TargetTriple) final {
    OS << BundleStartString << TargetTriple << "\n";
  }

  bool WriteBundleEnd(raw_fd_ostream &OS, StringRef TargetTriple) final {
    OS << BundleEndString << TargetTriple << "\n";
    return false;
  }

  void WriteBundle(raw_fd_ostream &OS, MemoryBuffer &Input) final {
    OS << Input.getBuffer();
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
  if (FilesType == "i")
    return new TextFileHandler(/*Comment=*/"//");
  if (FilesType == "ii")
    return new TextFileHandler(/*Comment=*/"//");
  if (FilesType == "ll")
    return new TextFileHandler(/*Comment=*/";");
  if (FilesType == "bc")
    return new BinaryFileHandler();
  if (FilesType == "s")
    return new TextFileHandler(/*Comment=*/"#");
  if (FilesType == "o")
    return CreateObjectFileHandler(FirstInput);
  if (FilesType == "gch")
    return new BinaryFileHandler();
  if (FilesType == "ast")
    return new BinaryFileHandler();

  errs() << "error: invalid file type specified.\n";
  return nullptr;
}

/// Bundle the files. Return true if an error was found.
static bool BundleFiles() {
  std::error_code EC;

  // Create output file.
  raw_fd_ostream OutputFile(OutputFileNames.front(), EC, sys::fs::F_None);

  if (EC) {
    errs() << "error: Can't open file " << OutputFileNames.front() << ".\n";
    return true;
  }

  // Open input files.
  std::vector<std::unique_ptr<MemoryBuffer>> InputBuffers(
      InputFileNames.size());

  unsigned Idx = 0;
  for (auto &I : InputFileNames) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
        MemoryBuffer::getFileOrSTDIN(I);
    if (std::error_code EC = CodeOrErr.getError()) {
      errs() << "error: Can't open file " << I << ": " << EC.message() << "\n";
      return true;
    }
    InputBuffers[Idx++] = std::move(CodeOrErr.get());
  }

  // Get the file handler. We use the host buffer as reference.
  assert(HostInputIndex != ~0u && "Host input index undefined??");
  std::unique_ptr<FileHandler> FH;
  FH.reset(CreateFileHandler(*InputBuffers[HostInputIndex].get()));

  // Quit if we don't have a handler.
  if (!FH.get())
    return true;

  // Write header.
  FH.get()->WriteHeader(OutputFile, InputBuffers);

  // Write all bundles along with the start/end markers. If an error was found
  // writing the end of the bundle component, abort the bundle writing.
  auto Input = InputBuffers.begin();
  for (auto &Triple : TargetNames) {
    FH.get()->WriteBundleStart(OutputFile, Triple);
    FH.get()->WriteBundle(OutputFile, *Input->get());
    if (FH.get()->WriteBundleEnd(OutputFile, Triple))
      return true;
    ++Input;
  }
  return false;
}

// Unbundle the files. Return true if an error was found.
static bool UnbundleFiles() {
  // Open Input file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> CodeOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFileNames.front());
  if (std::error_code EC = CodeOrErr.getError()) {
    errs() << "error: Can't open file " << InputFileNames.front() << ": "
           << EC.message() << "\n";
    return true;
  }

  MemoryBuffer &Input = *CodeOrErr.get();

  // Select the right files handler.
  std::unique_ptr<FileHandler> FH;
  FH.reset(CreateFileHandler(Input));

  // Quit if we don't have a handler.
  if (!FH.get())
    return true;

  // Read the header of the bundled file.
  FH.get()->ReadHeader(Input);

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
    StringRef CurTriple = FH.get()->ReadBundleStart(Input);

    // We don't have more bundles.
    if (CurTriple.empty())
      break;

    auto Output = Worklist.find(CurTriple);
    // The file may have more bundles for other targets, that we don't care
    // about. Therefore, move on to the next triple
    if (Output == Worklist.end()) {
      continue;
    }

    // Check if the output file can be opened and copy the bundle to it.
    std::error_code EC;
    raw_fd_ostream OutputFile(Output->second, EC, sys::fs::F_None);
    if (EC) {
      errs() << "error: Can't open file " << Output->second << ": "
             << EC.message() << "\n";
      return true;
    }
    FH.get()->ReadBundle(OutputFile, Input);
    FH.get()->ReadBundleEnd(Input);
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
      raw_fd_ostream OutputFile(E.second, EC, sys::fs::F_None);
      if (EC) {
        errs() << "error: Can't open file " << E.second << ": " << EC.message()
               << "\n";
        return true;
      }

      // If this entry has a host kind, copy the input file to the output file.
      if (hasHostKind(E.first()))
        OutputFile.write(Input.getBufferStart(), Input.getBufferSize());
    }
    return false;
  }

  // If we found elements, we emit an error if none of those were for the host.
  if (!FoundHostBundle) {
    errs() << "error: Can't find bundle for the host target\n";
    return true;
  }

  // If we still have any elements in the worklist, create empty files for them.
  for (auto &E : Worklist) {
    std::error_code EC;
    raw_fd_ostream OutputFile(E.second, EC, sys::fs::F_None);
    if (EC) {
      errs() << "error: Can't open file " << E.second << ": "  << EC.message()
             << "\n";
      return true;
    }
  }

  return false;
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

  bool Error = false;
  if (Unbundle) {
    if (InputFileNames.size() != 1) {
      Error = true;
      errs() << "error: only one input file supported in unbundling mode.\n";
    }
    if (OutputFileNames.size() != TargetNames.size()) {
      Error = true;
      errs() << "error: number of output files and targets should match in "
                "unbundling mode.\n";
    }
  } else {
    if (OutputFileNames.size() != 1) {
      Error = true;
      errs() << "error: only one output file supported in bundling mode.\n";
    }
    if (InputFileNames.size() != TargetNames.size()) {
      Error = true;
      errs() << "error: number of input files and targets should match in "
                "bundling mode.\n";
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
      errs() << "error: invalid target '" << Target << "'";

      if (!KindIsValid)
        errs() << ", unknown offloading kind '" << Kind << "'";
      if (!TripleIsValid)
        errs() << ", unknown target triple '" << Triple << "'";
      errs() << ".\n";
    }

    if (KindIsValid && Kind == "host") {
      ++HostTargetNum;
      // Save the index of the input that refers to the host.
      HostInputIndex = Index;
    }

    ++Index;
  }

  if (HostTargetNum != 1) {
    Error = true;
    errs() << "error: expecting exactly one host target but got "
           << HostTargetNum << ".\n";
  }

  if (Error)
    return 1;

  // Save the current executable directory as it will be useful to find other
  // tools.
  BundlerExecutable = sys::fs::getMainExecutable(argv[0], &BundlerExecutable);

  return Unbundle ? UnbundleFiles() : BundleFiles();
}
