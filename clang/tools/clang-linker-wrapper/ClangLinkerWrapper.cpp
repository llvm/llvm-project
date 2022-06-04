//===-- clang-linker-wrapper/ClangLinkerWrapper.cpp - wrapper over linker-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool works as a wrapper over a linking job. This tool is used to create
// linked device images for offloading. It scans the linker's input for embedded
// device offloading data stored in sections `.llvm.offloading.<triple>.<arch>`
// and extracts it as a temporary file. The extracted device files will then be
// passed to a device linking job to create a final device image.
//
//===---------------------------------------------------------------------===//

#include "OffloadWrapper.h"
#include "clang/Basic/Version.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/LTO.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

enum DebugKind {
  NoDebugInfo,
  DirectivesOnly,
  FullDebugInfo,
};

// Mark all our options with this category, everything else (except for -help)
// will be hidden.
static cl::OptionCategory
    ClangLinkerWrapperCategory("clang-linker-wrapper options");

static cl::opt<std::string> LinkerUserPath("linker-path", cl::Required,
                                           cl::desc("Path of linker binary"),
                                           cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string>
    TargetFeatures("target-feature",
                   cl::desc("Target features for triple"),
                   cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string> OptLevel("opt-level",
                                     cl::desc("Optimization level for LTO"),
                                     cl::init("O2"),
                                     cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    BitcodeLibraries("target-library", cl::ZeroOrMore,
                     cl::desc("Path for the target bitcode library"),
                     cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool> EmbedBitcode(
    "target-embed-bc",
    cl::desc("Embed linked bitcode instead of an executable device image"),
    cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool> DryRun(
    "dry-run",
    cl::desc("List the linker commands to be run without executing them"),
    cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool>
    PrintWrappedModule("print-wrapped-module",
                       cl::desc("Print the wrapped module's IR for testing"),
                       cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string>
    HostTriple("host-triple",
               cl::desc("Triple to use for the host compilation"),
               cl::init(sys::getDefaultTargetTriple()),
               cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    PtxasArgs("ptxas-args", cl::ZeroOrMore,
              cl::desc("Argument to pass to the ptxas invocation"),
              cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    LinkerArgs("device-linker", cl::ZeroOrMore,
               cl::desc("Arguments to pass to the device linker invocation"),
               cl::value_desc("<value> or <triple>=<value>"),
               cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool> Verbose("v",
                             cl::desc("Verbose output from tools"),
                             
                             cl::cat(ClangLinkerWrapperCategory));

static cl::opt<DebugKind> DebugInfo(
    cl::desc("Choose debugging level:"), cl::init(NoDebugInfo),
    cl::values(clEnumValN(NoDebugInfo, "g0", "No debug information"),
               clEnumValN(DirectivesOnly, "gline-directives-only",
                          "Direction information"),
               clEnumValN(FullDebugInfo, "g", "Full debugging support")));

static cl::opt<bool> SaveTemps("save-temps",
                               cl::desc("Save intermediary results."),
                               cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string> CudaPath("cuda-path",
                                     cl::desc("Save intermediary results."),
                                     cl::cat(ClangLinkerWrapperCategory));

// Do not parse linker options.
static cl::list<std::string>
    HostLinkerArgs(cl::Positional,
                   cl::desc("<options to be passed to linker>..."));

/// Path of the current binary.
static const char *LinkerExecutable;

/// Filename of the executable being created.
static StringRef ExecutableName;

/// System root if passed in to the linker via. '--sysroot='.
static StringRef Sysroot = "";

/// Binary path for the CUDA installation.
static std::string CudaBinaryPath;

/// Temporary files created by the linker wrapper.
static SmallVector<std::string, 16> TempFiles;

/// Codegen flags for LTO backend.
static codegen::RegisterCodeGenFlags CodeGenFlags;

/// Magic section string that marks the existence of offloading data. The
/// section will contain one or more offloading binaries stored contiguously.
#define OFFLOAD_SECTION_MAGIC_STR ".llvm.offloading"

/// The magic offset for the first object inside CUDA's fatbinary. This can be
/// different but it should work for what is passed here.
static constexpr unsigned FatbinaryOffset = 0x50;

/// Information for a device offloading file extracted from the host.
struct DeviceFile {
  DeviceFile(OffloadKind Kind, StringRef TheTriple, StringRef Arch,
             StringRef Filename)
      : Kind(Kind), TheTriple(TheTriple), Arch(Arch), Filename(Filename) {}

  OffloadKind Kind;
  std::string TheTriple;
  std::string Arch;
  std::string Filename;
};

namespace llvm {
/// Helper that allows DeviceFile to be used as a key in a DenseMap. For now we
/// assume device files with matching architectures and triples but different
/// offloading kinds should be handlded together, this may not be true in the
/// future.

// Provide DenseMapInfo for OffloadKind.
template <> struct DenseMapInfo<OffloadKind> {
  static inline OffloadKind getEmptyKey() { return OFK_LAST; }
  static inline OffloadKind getTombstoneKey() {
    return static_cast<OffloadKind>(OFK_LAST + 1);
  }
  static unsigned getHashValue(const OffloadKind &Val) { return Val * 37U; }

  static bool isEqual(const OffloadKind &LHS, const OffloadKind &RHS) {
    return LHS == RHS;
  }
};
template <> struct DenseMapInfo<DeviceFile> {
  static DeviceFile getEmptyKey() {
    return {DenseMapInfo<OffloadKind>::getEmptyKey(),
            DenseMapInfo<StringRef>::getEmptyKey(),
            DenseMapInfo<StringRef>::getEmptyKey(),
            DenseMapInfo<StringRef>::getEmptyKey()};
  }
  static DeviceFile getTombstoneKey() {
    return {DenseMapInfo<OffloadKind>::getTombstoneKey(),
            DenseMapInfo<StringRef>::getTombstoneKey(),
            DenseMapInfo<StringRef>::getTombstoneKey(),
            DenseMapInfo<StringRef>::getTombstoneKey()};
  }
  static unsigned getHashValue(const DeviceFile &I) {
    return DenseMapInfo<StringRef>::getHashValue(I.TheTriple) ^
           DenseMapInfo<StringRef>::getHashValue(I.Arch);
  }
  static bool isEqual(const DeviceFile &LHS, const DeviceFile &RHS) {
    return LHS.TheTriple == RHS.TheTriple && LHS.Arch == RHS.Arch;
  }
};
} // namespace llvm

namespace {

Error extractFromBuffer(std::unique_ptr<MemoryBuffer> Buffer,
                        SmallVectorImpl<DeviceFile> &DeviceFiles);

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  for (auto IC = std::next(CmdArgs.begin()), IE = CmdArgs.end(); IC != IE; ++IC)
    llvm::errs() << *IC << (std::next(IC) != IE ? " " : "\n");
}

// Forward user requested arguments to the device linking job.
void renderXLinkerArgs(SmallVectorImpl<StringRef> &Args, StringRef Triple) {
  for (StringRef Arg : LinkerArgs) {
    auto TripleAndValue = Arg.split('=');
    if (TripleAndValue.second.empty())
      Args.push_back(TripleAndValue.first);
    else if (TripleAndValue.first == Triple)
      Args.push_back(TripleAndValue.second);
  }
}

std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

/// Extract the device file from the string '<kind>-<triple>-<arch>=<library>'.
DeviceFile getBitcodeLibrary(StringRef LibraryStr) {
  auto DeviceAndPath = StringRef(LibraryStr).split('=');
  auto StringAndArch = DeviceAndPath.first.rsplit('-');
  auto KindAndTriple = StringAndArch.first.split('-');
  return DeviceFile(getOffloadKind(KindAndTriple.first), KindAndTriple.second,
                    StringAndArch.second, DeviceAndPath.second);
}

/// Get a temporary filename suitable for output.
Error createOutputFile(const Twine &Prefix, StringRef Extension,
                       SmallString<128> &NewFilename) {
  if (!SaveTemps) {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, NewFilename))
      return createFileError(NewFilename, EC);
    TempFiles.push_back(static_cast<std::string>(NewFilename));
  } else {
    const Twine &Filename = Prefix + "." + Extension;
    Filename.toNullTerminatedStringRef(NewFilename);
  }

  return Error::success();
}

/// Execute the command \p ExecutablePath with the arguments \p Args.
Error executeCommands(StringRef ExecutablePath, ArrayRef<StringRef> Args) {
  if (Verbose || DryRun)
    printCommands(Args);

  if (!DryRun)
    if (sys::ExecuteAndWait(ExecutablePath, Args))
      return createStringError(inconvertibleErrorCode(),
                               "'" + sys::path::filename(ExecutablePath) + "'" +
                                   " failed");
  return Error::success();
}

Expected<std::string> findProgram(StringRef Name, ArrayRef<StringRef> Paths) {

  ErrorOr<std::string> Path = sys::findProgramByName(Name, Paths);
  if (!Path)
    Path = sys::findProgramByName(Name);
  if (!Path && DryRun)
    return Name.str();
  if (!Path)
    return createStringError(Path.getError(),
                             "Unable to find '" + Name + "' in path");
  return *Path;
}

Error runLinker(std::string &LinkerPath, SmallVectorImpl<std::string> &Args) {
  std::vector<StringRef> LinkerArgs;
  LinkerArgs.push_back(LinkerPath);
  for (auto &Arg : Args)
    LinkerArgs.push_back(Arg);

  if (Error Err = executeCommands(LinkerPath, LinkerArgs))
    return Err;
  return Error::success();
}

void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-linker-wrapper") << '\n';
}

/// Attempts to extract all the embedded device images contained inside the
/// buffer \p Contents. The buffer is expected to contain a valid offloading
/// binary format.
Error extractOffloadFiles(StringRef Contents, StringRef Prefix,
                          SmallVectorImpl<DeviceFile> &DeviceFiles) {
  uint64_t Offset = 0;
  // There could be multiple offloading binaries stored at this section.
  while (Offset < Contents.size()) {
    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Contents.drop_front(Offset), "",
                                   /*RequiresNullTerminator*/ false);
    auto BinaryOrErr = OffloadBinary::create(*Buffer);
    if (!BinaryOrErr)
      return BinaryOrErr.takeError();
    OffloadBinary &Binary = **BinaryOrErr;

    if (Binary.getVersion() != 1)
      return createStringError(inconvertibleErrorCode(),
                               "Incompatible device image version");

    StringRef Kind = getOffloadKindName(Binary.getOffloadKind());
    StringRef Suffix = getImageKindName(Binary.getImageKind());

    SmallString<128> TempFile;
    if (Error Err =
            createOutputFile(Prefix + "-" + Kind + "-" + Binary.getTriple() +
                                 "-" + Binary.getArch(),
                             Suffix, TempFile))
      return Err;

    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(TempFile, Binary.getImage().size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    std::copy(Binary.getImage().bytes_begin(), Binary.getImage().bytes_end(),
              Output->getBufferStart());
    if (Error E = Output->commit())
      return E;

    DeviceFiles.emplace_back(Binary.getOffloadKind(), Binary.getTriple(),
                             Binary.getArch(), TempFile);

    Offset += Binary.getSize();
  }

  return Error::success();
}

Error extractFromBinary(const ObjectFile &Obj,
                        SmallVectorImpl<DeviceFile> &DeviceFiles) {
  StringRef Prefix = sys::path::stem(Obj.getFileName());

  // Extract offloading binaries from sections with the name `.llvm.offloading`.
  for (const SectionRef &Sec : Obj.sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (!Name || !Name->equals(OFFLOAD_SECTION_MAGIC_STR))
      continue;

    Expected<StringRef> Contents = Sec.getContents();
    if (!Contents)
      return Contents.takeError();

    if (Error Err = extractOffloadFiles(*Contents, Prefix, DeviceFiles))
      return Err;
  }

  return Error::success();
}

Error extractFromBitcode(std::unique_ptr<MemoryBuffer> Buffer,
                         SmallVectorImpl<DeviceFile> &DeviceFiles) {
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = getLazyIRModule(std::move(Buffer), Err, Context);
  if (!M)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create module");

  StringRef Prefix =
      sys::path::stem(M->getName()).take_until([](char C) { return C == '-'; });

  // Extract offloading data from globals with the `.llvm.offloading` section
  // name.
  for (GlobalVariable &GV : M->globals()) {
    if (!GV.hasSection() || !GV.getSection().equals(OFFLOAD_SECTION_MAGIC_STR))
      continue;

    auto *CDS = dyn_cast<ConstantDataSequential>(GV.getInitializer());
    if (!CDS)
      continue;

    StringRef Contents = CDS->getAsString();

    if (Error Err = extractOffloadFiles(Contents, Prefix, DeviceFiles))
      return Err;
  }

  return Error::success();
}

Error extractFromArchive(const Archive &Library,
                         SmallVectorImpl<DeviceFile> &DeviceFiles) {
  // Try to extract device code from each file stored in the static archive.
  Error Err = Error::success();
  for (auto Child : Library.children(Err)) {
    auto ChildBufferOrErr = Child.getMemoryBufferRef();
    if (!ChildBufferOrErr)
      return ChildBufferOrErr.takeError();
    std::unique_ptr<MemoryBuffer> ChildBuffer =
        MemoryBuffer::getMemBuffer(*ChildBufferOrErr, false);

    // Check if the buffer has the required alignment.
    if (!isAddrAligned(Align(OffloadBinary::getAlignment()),
                       ChildBuffer->getBufferStart()))
      ChildBuffer = MemoryBuffer::getMemBufferCopy(
          ChildBufferOrErr->getBuffer(),
          ChildBufferOrErr->getBufferIdentifier());

    if (Error Err = extractFromBuffer(std::move(ChildBuffer), DeviceFiles))
      return Err;
  }

  if (Err)
    return Err;
  return Error::success();
}

/// Extracts embedded device offloading code from a memory \p Buffer to a list
/// of \p DeviceFiles.
Error extractFromBuffer(std::unique_ptr<MemoryBuffer> Buffer,
                        SmallVectorImpl<DeviceFile> &DeviceFiles) {
  file_magic Type = identify_magic(Buffer->getBuffer());
  switch (Type) {
  case file_magic::bitcode:
    return extractFromBitcode(std::move(Buffer), DeviceFiles);
  case file_magic::elf_relocatable:
  case file_magic::macho_object:
  case file_magic::coff_object: {
    Expected<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(*Buffer, Type);
    if (!ObjFile)
      return ObjFile.takeError();
    return extractFromBinary(*ObjFile->get(), DeviceFiles);
  }
  case file_magic::archive: {
    Expected<std::unique_ptr<llvm::object::Archive>> LibFile =
        object::Archive::create(*Buffer);
    if (!LibFile)
      return LibFile.takeError();
    return extractFromArchive(*LibFile->get(), DeviceFiles);
  }
  default:
    return Error::success();
  }
}

// TODO: Move these to a separate file.
namespace nvptx {
Expected<std::string> assemble(StringRef InputFile, Triple TheTriple,
                               StringRef Arch, bool RDC = true) {
  // NVPTX uses the ptxas binary to create device object files.
  Expected<std::string> PtxasPath = findProgram("ptxas", {CudaBinaryPath});
  if (!PtxasPath)
    return PtxasPath.takeError();

  // Create a new file to write the linked device image to.
  SmallString<128> TempFile;
  if (Error Err =
          createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                               TheTriple.getArchName() + "-" + Arch,
                           "cubin", TempFile))
    return std::move(Err);

  SmallVector<StringRef, 16> CmdArgs;
  std::string Opt = "-" + OptLevel;
  CmdArgs.push_back(*PtxasPath);
  CmdArgs.push_back(TheTriple.isArch64Bit() ? "-m64" : "-m32");
  if (Verbose)
    CmdArgs.push_back("-v");
  if (DebugInfo == DirectivesOnly && OptLevel[1] == '0')
    CmdArgs.push_back("-lineinfo");
  else if (DebugInfo == FullDebugInfo && OptLevel[1] == '0')
    CmdArgs.push_back("-g");
  for (auto &Arg : PtxasArgs)
    CmdArgs.push_back(Arg);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(TempFile);
  CmdArgs.push_back(Opt);
  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Arch);
  if (RDC)
    CmdArgs.push_back("-c");

  CmdArgs.push_back(InputFile);

  if (Error Err = executeCommands(*PtxasPath, CmdArgs))
    return std::move(Err);

  return static_cast<std::string>(TempFile);
}

Expected<std::string> link(ArrayRef<std::string> InputFiles, Triple TheTriple,
                           StringRef Arch) {
  // NVPTX uses the nvlink binary to link device object files.
  Expected<std::string> NvlinkPath = findProgram("nvlink", {CudaBinaryPath});
  if (!NvlinkPath)
    return NvlinkPath.takeError();

  // Create a new file to write the linked device image to.
  SmallString<128> TempFile;
  if (Error Err =
          createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                               TheTriple.getArchName() + "-" + Arch,
                           "out", TempFile))
    return std::move(Err);

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*NvlinkPath);
  CmdArgs.push_back(TheTriple.isArch64Bit() ? "-m64" : "-m32");
  if (Verbose)
    CmdArgs.push_back("-v");
  if (DebugInfo != NoDebugInfo)
    CmdArgs.push_back("-g");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(TempFile);
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(Arch);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  renderXLinkerArgs(CmdArgs, TheTriple.getTriple());
  if (Error Err = executeCommands(*NvlinkPath, CmdArgs))
    return std::move(Err);

  return static_cast<std::string>(TempFile);
}

Expected<std::string> fatbinary(ArrayRef<StringRef> InputFiles,
                                Triple TheTriple, ArrayRef<StringRef> Archs) {
  // NVPTX uses the fatbinary program to bundle the linked images.
  Expected<std::string> FatBinaryPath =
      findProgram("fatbinary", {CudaBinaryPath});
  if (!FatBinaryPath)
    return FatBinaryPath.takeError();

  // Create a new file to write the linked device image to.
  SmallString<128> TempFile;
  if (Error Err = createOutputFile(sys::path::filename(ExecutableName) +
                                       "-device-" + TheTriple.getArchName(),
                                   "fatbin", TempFile))
    return std::move(Err);

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*FatBinaryPath);
  CmdArgs.push_back(TheTriple.isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back("--create");
  CmdArgs.push_back(TempFile);
  for (const auto &FileAndArch : llvm::zip(InputFiles, Archs))
    CmdArgs.push_back(Saver.save("--image=profile=" + std::get<1>(FileAndArch) +
                                 ",file=" + std::get<0>(FileAndArch)));

  if (Error Err = executeCommands(*FatBinaryPath, CmdArgs))
    return std::move(Err);

  return static_cast<std::string>(TempFile);
}
} // namespace nvptx
namespace amdgcn {
Expected<std::string> link(ArrayRef<std::string> InputFiles, Triple TheTriple,
                           StringRef Arch) {
  // AMDGPU uses lld to link device object files.
  Expected<std::string> LLDPath =
      findProgram("lld", {getMainExecutable("lld")});
  if (!LLDPath)
    return LLDPath.takeError();

  // Create a new file to write the linked device image to.
  SmallString<128> TempFile;
  if (Error Err = createOutputFile(sys::path::filename(ExecutableName) + "-" +
                                       TheTriple.getArchName() + "-" + Arch,
                                   "out", TempFile))
    return std::move(Err);

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*LLDPath);
  CmdArgs.push_back("-flavor");
  CmdArgs.push_back("gnu");
  CmdArgs.push_back("--no-undefined");
  CmdArgs.push_back("-shared");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(TempFile);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  renderXLinkerArgs(CmdArgs, TheTriple.getTriple());
  if (Error Err = executeCommands(*LLDPath, CmdArgs))
    return std::move(Err);

  return static_cast<std::string>(TempFile);
}
} // namespace amdgcn

namespace generic {

const char *getLDMOption(const llvm::Triple &T) {
  switch (T.getArch()) {
  case llvm::Triple::x86:
    if (T.isOSIAMCU())
      return "elf_iamcu";
    return "elf_i386";
  case llvm::Triple::aarch64:
    return "aarch64linux";
  case llvm::Triple::aarch64_be:
    return "aarch64linuxb";
  case llvm::Triple::ppc64:
    return "elf64ppc";
  case llvm::Triple::ppc64le:
    return "elf64lppc";
  case llvm::Triple::x86_64:
    if (T.isX32())
      return "elf32_x86_64";
    return "elf_x86_64";
  case llvm::Triple::ve:
    return "elf64ve";
  default:
    return nullptr;
  }
}

Expected<std::string> link(ArrayRef<std::string> InputFiles, Triple TheTriple,
                           StringRef Arch) {
  // Create a new file to write the linked device image to.
  SmallString<128> TempFile;
  if (Error Err = createOutputFile(sys::path::filename(ExecutableName) + "-" +
                                       TheTriple.getArchName() + "-" + Arch,
                                   "out", TempFile))
    return std::move(Err);

  // Use the host linker to perform generic offloading. Use the same libraries
  // and paths as the host application does.
  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(LinkerUserPath);
  CmdArgs.push_back("-m");
  CmdArgs.push_back(getLDMOption(TheTriple));
  CmdArgs.push_back("-shared");
  for (auto AI = HostLinkerArgs.begin(), AE = HostLinkerArgs.end(); AI != AE;
       ++AI) {
    StringRef Arg = *AI;
    if (Arg.startswith("-L"))
      CmdArgs.push_back(Arg);
    else if (Arg.startswith("-l"))
      CmdArgs.push_back(Arg);
    else if (Arg.startswith("--as-needed"))
      CmdArgs.push_back(Arg);
    else if (Arg.startswith("--no-as-needed"))
      CmdArgs.push_back(Arg);
    else if (Arg.startswith("-rpath")) {
      CmdArgs.push_back(Arg);
      CmdArgs.push_back(*std::next(AI));
    } else if (Arg.startswith("-dynamic-linker")) {
      CmdArgs.push_back(Arg);
      CmdArgs.push_back(*std::next(AI));
    }
  }
  CmdArgs.push_back("-Bsymbolic");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(TempFile);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  renderXLinkerArgs(CmdArgs, TheTriple.getTriple());
  if (Error Err = executeCommands(LinkerUserPath, CmdArgs))
    return std::move(Err);

  return static_cast<std::string>(TempFile);
}
} // namespace generic

Expected<std::string> linkDevice(ArrayRef<std::string> InputFiles,
                                 Triple TheTriple, StringRef Arch) {
  switch (TheTriple.getArch()) {
  case Triple::nvptx:
  case Triple::nvptx64:
    return nvptx::link(InputFiles, TheTriple, Arch);
  case Triple::amdgcn:
    return amdgcn::link(InputFiles, TheTriple, Arch);
  case Triple::x86:
  case Triple::x86_64:
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::ppc64:
  case Triple::ppc64le:
    return generic::link(InputFiles, TheTriple, Arch);
  default:
    return createStringError(inconvertibleErrorCode(),
                             TheTriple.getArchName() +
                                 " linking is not supported");
  }
}

void diagnosticHandler(const DiagnosticInfo &DI) {
  std::string ErrStorage;
  raw_string_ostream OS(ErrStorage);
  DiagnosticPrinterRawOStream DP(OS);
  DI.print(DP);

  switch (DI.getSeverity()) {
  case DS_Error:
    WithColor::error(errs(), LinkerExecutable) << ErrStorage << "\n";
    break;
  case DS_Warning:
    WithColor::warning(errs(), LinkerExecutable) << ErrStorage << "\n";
    break;
  case DS_Note:
    WithColor::note(errs(), LinkerExecutable) << ErrStorage << "\n";
    break;
  case DS_Remark:
    WithColor::remark(errs()) << ErrStorage << "\n";
    break;
  }
}

// Get the target features passed in from the driver as <triple>=<features>.
std::vector<std::string> getTargetFeatures(const Triple &TheTriple) {
  std::vector<std::string> Features;
  auto TargetAndFeatures = StringRef(TargetFeatures).split('=');
  if (TargetAndFeatures.first != TheTriple.getTriple())
    return Features;

  for (auto Feature : llvm::split(TargetAndFeatures.second, ','))
    Features.push_back(Feature.str());
  return Features;
}

CodeGenOpt::Level getCGOptLevel(unsigned OptLevel) {
  switch (OptLevel) {
  case 0:
    return CodeGenOpt::None;
  case 1:
    return CodeGenOpt::Less;
  case 2:
    return CodeGenOpt::Default;
  case 3:
    return CodeGenOpt::Aggressive;
  }
  llvm_unreachable("Invalid optimization level");
}

template <typename ModuleHook = function_ref<bool(size_t, const Module &)>>
std::unique_ptr<lto::LTO> createLTO(
    const Triple &TheTriple, StringRef Arch, bool WholeProgram,
    ModuleHook Hook = [](size_t, const Module &) { return true; }) {
  lto::Config Conf;
  lto::ThinBackend Backend;
  // TODO: Handle index-only thin-LTO
  Backend =
      lto::createInProcessThinBackend(llvm::heavyweight_hardware_concurrency());

  Conf.CPU = Arch.str();
  Conf.Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);

  Conf.MAttrs = getTargetFeatures(TheTriple);
  Conf.CGOptLevel = getCGOptLevel(OptLevel[1] - '0');
  Conf.OptLevel = OptLevel[1] - '0';
  if (Conf.OptLevel > 0)
    Conf.UseDefaultPipeline = true;
  Conf.DefaultTriple = TheTriple.getTriple();
  Conf.DiagHandler = diagnosticHandler;

  Conf.PTO.LoopVectorization = Conf.OptLevel > 1;
  Conf.PTO.SLPVectorization = Conf.OptLevel > 1;

  if (SaveTemps) {
    auto HandleError = [&](Error Err) {
      logAllUnhandledErrors(std::move(Err),
                            WithColor::error(errs(), LinkerExecutable));
      exit(1);
    };
    Conf.PostInternalizeModuleHook = [&](size_t, const Module &M) {
      SmallString<128> TempFile;
      if (Error Err = createOutputFile(sys::path::filename(ExecutableName) +
                                           "-device-" + TheTriple.getTriple(),
                                       "bc", TempFile))
        HandleError(std::move(Err));

      std::error_code EC;
      raw_fd_ostream LinkedBitcode(TempFile, EC, sys::fs::OF_None);
      if (EC)
        HandleError(errorCodeToError(EC));
      WriteBitcodeToFile(M, LinkedBitcode);
      return true;
    };
  }
  Conf.PostOptModuleHook = Hook;
  if (TheTriple.isNVPTX())
    Conf.CGFileType = CGFT_AssemblyFile;
  else
    Conf.CGFileType = CGFT_ObjectFile;

  // TODO: Handle remark files
  Conf.HasWholeProgramVisibility = WholeProgram;

  return std::make_unique<lto::LTO>(std::move(Conf), Backend);
}

// Returns true if \p S is valid as a C language identifier and will be given
// `__start_` and `__stop_` symbols.
bool isValidCIdentifier(StringRef S) {
  return !S.empty() && (isAlpha(S[0]) || S[0] == '_') &&
         std::all_of(S.begin() + 1, S.end(),
                     [](char C) { return C == '_' || isAlnum(C); });
}

Error linkBitcodeFiles(SmallVectorImpl<std::string> &InputFiles,
                       const Triple &TheTriple, StringRef Arch,
                       bool &WholeProgram) {
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> SavedBuffers;
  SmallVector<std::unique_ptr<lto::InputFile>, 4> BitcodeFiles;
  SmallVector<std::string, 4> NewInputFiles;
  DenseSet<StringRef> UsedInRegularObj;
  DenseSet<StringRef> UsedInSharedLib;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  // Search for bitcode files in the input and create an LTO input file. If it
  // is not a bitcode file, scan its symbol table for symbols we need to
  // save.
  for (StringRef File : InputFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(File);
    if (std::error_code EC = BufferOrErr.getError())
      return createFileError(File, EC);
    MemoryBufferRef Buffer = **BufferOrErr;

    file_magic Type = identify_magic((*BufferOrErr)->getBuffer());
    switch (Type) {
    case file_magic::bitcode: {
      Expected<std::unique_ptr<lto::InputFile>> InputFileOrErr =
          llvm::lto::InputFile::create(Buffer);
      if (!InputFileOrErr)
        return InputFileOrErr.takeError();

      // Save the input file and the buffer associated with its memory.
      BitcodeFiles.push_back(std::move(*InputFileOrErr));
      SavedBuffers.push_back(std::move(*BufferOrErr));
      continue;
    }
    case file_magic::cuda_fatbinary: {
      // Cuda fatbinaries made by Clang almost almost have an object eighty
      // bytes from the beginning. This should be sufficient to identify the
      // symbols.
      Buffer = MemoryBufferRef(
          (*BufferOrErr)->getBuffer().drop_front(FatbinaryOffset), "FatBinary");
      LLVM_FALLTHROUGH;
    }
    case file_magic::elf_relocatable:
    case file_magic::elf_shared_object:
    case file_magic::macho_object:
    case file_magic::coff_object: {
      Expected<std::unique_ptr<ObjectFile>> ObjFile =
          ObjectFile::createObjectFile(Buffer);
      if (!ObjFile)
        continue;

      NewInputFiles.push_back(File.str());
      for (auto &Sym : (*ObjFile)->symbols()) {
        Expected<StringRef> Name = Sym.getName();
        if (!Name)
          return Name.takeError();

        // Record if we've seen these symbols in any object or shared libraries.
        if ((*ObjFile)->isRelocatableObject())
          UsedInRegularObj.insert(Saver.save(*Name));
        else
          UsedInSharedLib.insert(Saver.save(*Name));
      }
      continue;
    }
    default:
      continue;
    }
  }

  if (BitcodeFiles.empty())
    return Error::success();

  auto HandleError = [&](Error Err) {
    logAllUnhandledErrors(std::move(Err),
                          WithColor::error(errs(), LinkerExecutable));
    exit(1);
  };

  // LTO Module hook to output bitcode without running the backend.
  auto OutputBitcode = [&](size_t Task, const Module &M) {
    SmallString<128> TempFile;
    if (Error Err = createOutputFile(sys::path::filename(ExecutableName) +
                                         "-jit-" + TheTriple.getTriple(),
                                     "bc", TempFile))
      HandleError(std::move(Err));

    std::error_code EC;
    raw_fd_ostream LinkedBitcode(TempFile, EC, sys::fs::OF_None);
    if (EC)
      HandleError(errorCodeToError(EC));
    WriteBitcodeToFile(M, LinkedBitcode);
    NewInputFiles.push_back(static_cast<std::string>(TempFile));
    return false;
  };

  // We assume visibility of the whole program if every input file was bitcode.
  WholeProgram = BitcodeFiles.size() == InputFiles.size();
  auto LTOBackend =
      (EmbedBitcode) ? createLTO(TheTriple, Arch, WholeProgram, OutputBitcode)
                     : createLTO(TheTriple, Arch, WholeProgram);

  // We need to resolve the symbols so the LTO backend knows which symbols need
  // to be kept or can be internalized. This is a simplified symbol resolution
  // scheme to approximate the full resolution a linker would do.
  DenseSet<StringRef> PrevailingSymbols;
  for (auto &BitcodeFile : BitcodeFiles) {
    const auto Symbols = BitcodeFile->symbols();
    SmallVector<lto::SymbolResolution, 16> Resolutions(Symbols.size());
    size_t Idx = 0;
    for (auto &Sym : Symbols) {
      lto::SymbolResolution &Res = Resolutions[Idx++];

      // We will use this as the prevailing symbol definition in LTO unless
      // it is undefined or another definition has already been used.
      Res.Prevailing =
          !Sym.isUndefined() &&
          PrevailingSymbols.insert(Saver.save(Sym.getName())).second;

      // We need LTO to preseve the following global symbols:
      // 1) Symbols used in regular objects.
      // 2) Sections that will be given a __start/__stop symbol.
      // 3) Prevailing symbols that are needed visible to external libraries.
      Res.VisibleToRegularObj =
          UsedInRegularObj.contains(Sym.getName()) ||
          isValidCIdentifier(Sym.getSectionName()) ||
          (Res.Prevailing &&
           (Sym.getVisibility() != GlobalValue::HiddenVisibility &&
            !Sym.canBeOmittedFromSymbolTable()));

      // Identify symbols that must be exported dynamically and can be
      // referenced by other files.
      Res.ExportDynamic =
          Sym.getVisibility() != GlobalValue::HiddenVisibility &&
          (UsedInSharedLib.contains(Sym.getName()) ||
           !Sym.canBeOmittedFromSymbolTable());

      // The final definition will reside in this linkage unit if the symbol is
      // defined and local to the module. This only checks for bitcode files,
      // full assertion will require complete symbol resolution.
      Res.FinalDefinitionInLinkageUnit =
          Sym.getVisibility() != GlobalValue::DefaultVisibility &&
          (!Sym.isUndefined() && !Sym.isCommon());

      // We do not support linker redefined symbols (e.g. --wrap) for device
      // image linking, so the symbols will not be changed after LTO.
      Res.LinkerRedefined = false;
    }

    // Add the bitcode file with its resolved symbols to the LTO job.
    if (Error Err = LTOBackend->add(std::move(BitcodeFile), Resolutions))
      return Err;
  }

  // Run the LTO job to compile the bitcode.
  size_t MaxTasks = LTOBackend->getMaxTasks();
  std::vector<SmallString<128>> Files(MaxTasks);
  auto AddStream = [&](size_t Task) -> std::unique_ptr<CachedFileStream> {
    int FD = -1;
    auto &TempFile = Files[Task];
    StringRef Extension = (TheTriple.isNVPTX()) ? "s" : "o";
    if (Error Err = createOutputFile(sys::path::filename(ExecutableName) +
                                         "-device-" + TheTriple.getTriple(),
                                     Extension, TempFile))
      HandleError(std::move(Err));
    if (std::error_code EC = sys::fs::openFileForWrite(TempFile, FD))
      HandleError(errorCodeToError(EC));
    return std::make_unique<CachedFileStream>(
        std::make_unique<llvm::raw_fd_ostream>(FD, true));
  };

  if (Error Err = LTOBackend->run(AddStream))
    return Err;

  // If we are embedding bitcode we only need the intermediate output.
  if (EmbedBitcode) {
    InputFiles = NewInputFiles;
    return Error::success();
  }

  // Is we are compiling for NVPTX we need to run the assembler first.
  if (TheTriple.isNVPTX()) {
    for (auto &File : Files) {
      auto FileOrErr = nvptx::assemble(File, TheTriple, Arch, !WholeProgram);
      if (!FileOrErr)
        return FileOrErr.takeError();
      File = *FileOrErr;
    }
  }

  // Append the new inputs to the device linker input.
  for (auto &File : Files)
    NewInputFiles.push_back(static_cast<std::string>(File));
  InputFiles = NewInputFiles;

  return Error::success();
}

/// Runs the appropriate linking action on all the device files specified in \p
/// DeviceFiles. The linked device images are returned in \p LinkedImages.
Error linkDeviceFiles(ArrayRef<DeviceFile> DeviceFiles,
                      ArrayRef<DeviceFile> LibraryFiles,
                      SmallVectorImpl<DeviceFile> &LinkedImages) {
  // Get the list of inputs and active offload kinds for a specific device.
  DenseMap<DeviceFile, SmallVector<std::string, 4>> LinkerInputMap;
  DenseMap<DeviceFile, DenseSet<OffloadKind>> ActiveOffloadKinds;
  for (auto &File : DeviceFiles) {
    LinkerInputMap[File].push_back(File.Filename);
    ActiveOffloadKinds[File].insert(File.Kind);
  }

  // Static libraries are loaded lazily as-needed, only add them if other files
  // are present.
  // TODO: We need to check the symbols as well, static libraries are only
  //       loaded if they contain symbols that are currently undefined or common
  //       in the symbol table.
  for (auto &File : LibraryFiles)
    if (LinkerInputMap.count(File))
      LinkerInputMap[File].push_back(File.Filename);

  // Try to link each device toolchain.
  for (auto &LinkerInput : LinkerInputMap) {
    DeviceFile &File = LinkerInput.getFirst();
    Triple TheTriple = Triple(File.TheTriple);
    auto &LinkerInputFiles = LinkerInput.getSecond();
    bool WholeProgram = false;

    // Run LTO on any bitcode files and replace the input with the result.
    if (Error Err = linkBitcodeFiles(LinkerInputFiles, TheTriple, File.Arch,
                                     WholeProgram))
      return Err;

    if (EmbedBitcode) {
      // If we are embedding bitcode for JIT, skip the final device linking.
      if (LinkerInputFiles.size() != 1 || !WholeProgram)
        return createStringError(inconvertibleErrorCode(),
                                 "Unable to embed bitcode image for JIT");
      LinkedImages.emplace_back(OFK_OpenMP, TheTriple.getTriple(), File.Arch,
                                LinkerInputFiles.front());
      continue;
    }
    if (WholeProgram && TheTriple.isNVPTX()) {
      // If we performed LTO on NVPTX and had whole program visibility, we can
      // use CUDA in non-RDC mode.
      if (LinkerInputFiles.size() != 1)
        return createStringError(inconvertibleErrorCode(),
                                 "Invalid number of inputs for non-RDC mode");
      for (OffloadKind Kind : ActiveOffloadKinds[LinkerInput.getFirst()])
        LinkedImages.emplace_back(Kind, TheTriple.getTriple(), File.Arch,
                                  LinkerInputFiles.front());
      continue;
    }

    auto ImageOrErr = linkDevice(LinkerInputFiles, TheTriple, File.Arch);
    if (!ImageOrErr)
      return ImageOrErr.takeError();

    // Create separate images for all the active offload kinds.
    for (OffloadKind Kind : ActiveOffloadKinds[LinkerInput.getFirst()])
      LinkedImages.emplace_back(Kind, TheTriple.getTriple(), File.Arch,
                                *ImageOrErr);
  }
  return Error::success();
}

// Compile the module to an object file using the appropriate target machine for
// the host triple.
Expected<std::string> compileModule(Module &M) {
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return createStringError(inconvertibleErrorCode(), Msg);

  auto Options =
      codegen::InitTargetOptionsFromCodeGenFlags(Triple(M.getTargetTriple()));
  StringRef CPU = "";
  StringRef Features = "";
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      HostTriple, CPU, Features, Options, Reloc::PIC_, M.getCodeModel()));

  if (M.getDataLayout().isDefault())
    M.setDataLayout(TM->createDataLayout());

  SmallString<128> ObjectFile;
  int FD = -1;
  if (Error Err = createOutputFile(
          sys::path::filename(ExecutableName) + "-wrapper", "o", ObjectFile))
    return std::move(Err);
  if (std::error_code EC = sys::fs::openFileForWrite(ObjectFile, FD))
    return errorCodeToError(EC);

  auto OS = std::make_unique<llvm::raw_fd_ostream>(FD, true);

  legacy::PassManager CodeGenPasses;
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS, nullptr, CGFT_ObjectFile))
    return createStringError(inconvertibleErrorCode(),
                             "Failed to execute host backend");
  CodeGenPasses.run(M);

  return static_cast<std::string>(ObjectFile);
}

/// Load all of the OpenMP images into a buffer and pass it to the binary
/// wrapping function to create the registration code in the module \p M.
Error wrapOpenMPImages(Module &M, ArrayRef<DeviceFile> Images) {
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> SavedBuffers;
  SmallVector<ArrayRef<char>, 4> ImagesToWrap;
  for (const DeviceFile &File : Images) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
        llvm::MemoryBuffer::getFileOrSTDIN(File.Filename);
    if (std::error_code EC = ImageOrError.getError())
      return createFileError(File.Filename, EC);
    ImagesToWrap.emplace_back((*ImageOrError)->getBufferStart(),
                              (*ImageOrError)->getBufferSize());
    SavedBuffers.emplace_back(std::move(*ImageOrError));
  }

  if (Error Err = wrapOpenMPBinaries(M, ImagesToWrap))
    return Err;
  return Error::success();
}

/// Combine all of the CUDA images into a single fatbinary and pass it to the
/// binary wrapping function to create the registration code in the module \p M.
Error wrapCudaImages(Module &M, ArrayRef<DeviceFile> Images) {
  SmallVector<StringRef, 4> InputFiles;
  SmallVector<StringRef, 4> Architectures;
  for (const DeviceFile &File : Images) {
    InputFiles.push_back(File.Filename);
    Architectures.push_back(File.Arch);
  }

  // CUDA expects its embedded device images to be a fatbinary.
  Triple TheTriple = Triple(Images.front().TheTriple);
  auto FileOrErr = nvptx::fatbinary(InputFiles, TheTriple, Architectures);
  if (!FileOrErr)
    return FileOrErr.takeError();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(*FileOrErr);
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(*FileOrErr, EC);

  auto ImageToWrap = ArrayRef<char>((*ImageOrError)->getBufferStart(),
                                    (*ImageOrError)->getBufferSize());

  if (Error Err = wrapCudaBinary(M, ImageToWrap))
    return Err;
  return Error::success();
}

/// Creates the object file containing the device image and runtime
/// registration code from the device images stored in \p Images.
Expected<SmallVector<std::string, 2>>
wrapDeviceImages(ArrayRef<DeviceFile> Images) {
  DenseMap<OffloadKind, SmallVector<DeviceFile, 2>> ImagesForKind;
  for (const DeviceFile &Image : Images)
    ImagesForKind[Image.Kind].push_back(Image);

  SmallVector<std::string, 2> WrappedImages;
  for (const auto &KindAndImages : ImagesForKind) {
    LLVMContext Context;
    Module M("offload.wrapper.module", Context);
    M.setTargetTriple(HostTriple);

    // Create registration code for the given offload kinds in the Module.
    switch (KindAndImages.getFirst()) {
    case OFK_OpenMP:
      if (Error Err = wrapOpenMPImages(M, KindAndImages.getSecond()))
        return std::move(Err);
      break;
    case OFK_Cuda:
      if (Error Err = wrapCudaImages(M, KindAndImages.getSecond()))
        return std::move(Err);
      break;
    default:
      return createStringError(inconvertibleErrorCode(),
                               getOffloadKindName(KindAndImages.getFirst()) +
                                   " wrapping is not supported");
    }

    if (PrintWrappedModule)
      llvm::errs() << M;

    auto FileOrErr = compileModule(M);
    if (!FileOrErr)
      return FileOrErr.takeError();
    WrappedImages.push_back(*FileOrErr);
  }

  return WrappedImages;
}

Optional<std::string> findFile(StringRef Dir, const Twine &Name) {
  SmallString<128> Path;
  if (Dir.startswith("="))
    sys::path::append(Path, Sysroot, Dir.substr(1), Name);
  else
    sys::path::append(Path, Dir, Name);

  if (sys::fs::exists(Path))
    return static_cast<std::string>(Path);
  return None;
}

Optional<std::string> findFromSearchPaths(StringRef Name,
                                          ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (Optional<std::string> File = findFile(Dir, Name))
      return File;
  return None;
}

Optional<std::string> searchLibraryBaseName(StringRef Name,
                                            ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths) {
    if (Optional<std::string> File = findFile(Dir, "lib" + Name + ".so"))
      return None;
    if (Optional<std::string> File = findFile(Dir, "lib" + Name + ".a"))
      return File;
  }
  return None;
}

/// Search for static libraries in the linker's library path given input like
/// `-lfoo` or `-l:libfoo.a`.
Optional<std::string> searchLibrary(StringRef Input,
                                    ArrayRef<StringRef> SearchPaths) {
  if (!Input.startswith("-l"))
    return None;
  StringRef Name = Input.drop_front(2);
  if (Name.startswith(":"))
    return findFromSearchPaths(Name.drop_front(), SearchPaths);
  return searchLibraryBaseName(Name, SearchPaths);
}

} // namespace

int main(int argc, const char **argv) {
  InitLLVM X(argc, argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  LinkerExecutable = argv[0];
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::SetVersionPrinter(PrintVersion);
  cl::HideUnrelatedOptions(ClangLinkerWrapperCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A wrapper utility over the host linker. It scans the input files for\n"
      "sections that require additional processing prior to linking. The tool\n"
      "will then transparently pass all arguments and input to the specified\n"
      "host linker to create the final binary.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    return EXIT_FAILURE;
  };

  if (!CudaPath.empty())
    CudaBinaryPath = CudaPath + "/bin";

  auto RootIt = llvm::find_if(HostLinkerArgs, [](StringRef Arg) {
    return Arg.startswith("--sysroot=");
  });
  if (RootIt != HostLinkerArgs.end())
    Sysroot = StringRef(*RootIt).split('=').second;

  ExecutableName = *std::next(llvm::find(HostLinkerArgs, "-o"));
  SmallVector<std::string, 16> LinkerArgs;
  for (const std::string &Arg : HostLinkerArgs)
    LinkerArgs.push_back(Arg);

  SmallVector<StringRef, 16> LibraryPaths;
  for (StringRef Arg : LinkerArgs) {
    if (Arg.startswith("-L"))
      LibraryPaths.push_back(Arg.drop_front(2));
  }

  // Try to extract device code from the linker input.
  SmallVector<DeviceFile, 4> DeviceFiles;
  SmallVector<DeviceFile, 4> LibraryFiles;
  for (StringRef Arg : LinkerArgs) {
    if (Arg == ExecutableName)
      continue;

    // Search the inpuot argument for embedded device files if it is a static
    // library or regular input file.
    if (Optional<std::string> Library = searchLibrary(Arg, LibraryPaths)) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
          MemoryBuffer::getFileOrSTDIN(*Library);
      if (std::error_code EC = BufferOrErr.getError())
        return reportError(createFileError(*Library, EC));

      if (Error Err = extractFromBuffer(std::move(*BufferOrErr), LibraryFiles))
        return reportError(std::move(Err));
    } else if (sys::fs::exists(Arg) && !sys::fs::is_directory(Arg)) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
          MemoryBuffer::getFileOrSTDIN(Arg);
      if (std::error_code EC = BufferOrErr.getError())
        return reportError(createFileError(Arg, EC));

      if (Error Err = extractFromBuffer(std::move(*BufferOrErr), DeviceFiles))
        return reportError(std::move(Err));
    }
  }

  // Add the device bitcode libraries to the device files if any were passed in.
  for (StringRef LibraryStr : BitcodeLibraries)
    DeviceFiles.push_back(getBitcodeLibrary(LibraryStr));

  // Link the device images extracted from the linker input.
  SmallVector<DeviceFile, 4> LinkedImages;
  if (Error Err = linkDeviceFiles(DeviceFiles, LibraryFiles, LinkedImages))
    return reportError(std::move(Err));

  // Wrap each linked device image into a linkable host binary and add it to the
  // link job's inputs.
  auto FileOrErr = wrapDeviceImages(LinkedImages);
  if (!FileOrErr)
    return reportError(FileOrErr.takeError());

  // We need to insert the new files next to the old ones to make sure they're
  // linked with the same libraries / arguments.
  if (!FileOrErr->empty()) {
    auto *FirstInput = std::next(llvm::find_if(LinkerArgs, [](StringRef Str) {
      return sys::fs::exists(Str) && !sys::fs::is_directory(Str) &&
             Str != ExecutableName;
    }));
    LinkerArgs.insert(FirstInput, FileOrErr->begin(), FileOrErr->end());
  }

  // Run the host linking job.
  if (Error Err = runLinker(LinkerUserPath, LinkerArgs))
    return reportError(std::move(Err));

  // Remove the temporary files created.
  for (const auto &TempFile : TempFiles)
    if (std::error_code EC = sys::fs::remove(TempFile))
      reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
