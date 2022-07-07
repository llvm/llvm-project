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
// device offloading data stored in sections `.llvm.offloading` and extracts it
// as a temporary file. The extracted device files will then be passed to a
// device linking job to create a final device image.
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
#include "llvm/Object/IRObjectFile.h"
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

static cl::opt<std::string> OptLevel("opt-level",
                                     cl::desc("Optimization level for LTO"),
                                     cl::init("O2"),
                                     cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    BitcodeLibraries("target-library",
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
    PtxasArgs("ptxas-args",
              cl::desc("Argument to pass to the ptxas invocation"),
              cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    LinkerArgs("device-linker",
               cl::desc("Arguments to pass to the device linker invocation"),
               cl::value_desc("<value> or <triple>=<value>"),
               cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool> Verbose("v", cl::desc("Verbose output from tools"),

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
static std::list<SmallString<128>> TempFiles;

/// Codegen flags for LTO backend.
static codegen::RegisterCodeGenFlags CodeGenFlags;

/// Magic section string that marks the existence of offloading data. The
/// section will contain one or more offloading binaries stored contiguously.
#define OFFLOAD_SECTION_MAGIC_STR ".llvm.offloading"

/// The magic offset for the first object inside CUDA's fatbinary. This can be
/// different but it should work for what is passed here.
static constexpr unsigned FatbinaryOffset = 0x50;

using OffloadingImage = OffloadBinary::OffloadingImage;

/// A class to contain the binary information for a single OffloadBinary.
class OffloadFile : public OwningBinary<OffloadBinary> {
public:
  using TargetID = std::pair<StringRef, StringRef>;

  OffloadFile(std::unique_ptr<OffloadBinary> Binary,
              std::unique_ptr<MemoryBuffer> Buffer)
      : OwningBinary<OffloadBinary>(std::move(Binary), std::move(Buffer)) {}

  /// We use the Triple and Architecture pair to group linker inputs together.
  /// This conversion function lets us use these files in a hash-map.
  operator TargetID() const {
    return std::make_pair(getBinary()->getTriple(), getBinary()->getArch());
  }
};

namespace llvm {
// Provide DenseMapInfo so that OffloadKind can be used in a DenseMap.
template <> struct DenseMapInfo<OffloadKind> {
  static inline OffloadKind getEmptyKey() { return OFK_LAST; }
  static inline OffloadKind getTombstoneKey() {
    return static_cast<OffloadKind>(OFK_LAST + 1);
  }
  static unsigned getHashValue(const OffloadKind &Val) { return Val; }

  static bool isEqual(const OffloadKind &LHS, const OffloadKind &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace {

Error extractFromBuffer(std::unique_ptr<MemoryBuffer> Buffer,
                        SmallVectorImpl<OffloadFile> &DeviceFiles);

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  for (auto IC = std::next(CmdArgs.begin()), IE = CmdArgs.end(); IC != IE; ++IC)
    llvm::errs() << *IC << (std::next(IC) != IE ? " " : "\n");
}

/// Forward user requested arguments to the device linking job.
void renderXLinkerArgs(SmallVectorImpl<StringRef> &Args, StringRef Triple) {
  for (StringRef Arg : LinkerArgs) {
    auto TripleAndValue = Arg.split('=');
    if (TripleAndValue.second.empty())
      Args.push_back(TripleAndValue.first);
    else if (TripleAndValue.first == Triple)
      Args.push_back(TripleAndValue.second);
  }
}

/// Create an extra user-specified \p OffloadFile.
/// TODO: We should find a way to wrap these as libraries instead.
Expected<OffloadFile> getInputBitcodeLibrary(StringRef Input) {
  auto DeviceAndPath = StringRef(Input).split('=');
  auto StringAndArch = DeviceAndPath.first.rsplit('-');
  auto KindAndTriple = StringAndArch.first.split('-');

  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(DeviceAndPath.second);
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(DeviceAndPath.second, EC);

  OffloadingImage Image{};
  Image.TheImageKind = IMG_Bitcode;
  Image.TheOffloadKind = getOffloadKind(KindAndTriple.first);
  Image.StringData = {{"triple", KindAndTriple.second},
                      {"arch", StringAndArch.second}};
  Image.Image = std::move(*ImageOrError);

  std::unique_ptr<MemoryBuffer> Binary = OffloadBinary::write(Image);
  auto NewBinaryOrErr = OffloadBinary::create(*Binary);
  if (!NewBinaryOrErr)
    return NewBinaryOrErr.takeError();
  return OffloadFile(std::move(*NewBinaryOrErr), std::move(Binary));
}

std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

/// Get a temporary filename suitable for output.
Expected<StringRef> createOutputFile(const Twine &Prefix, StringRef Extension) {
  SmallString<128> OutputFile;
  if (SaveTemps) {
    (Prefix + "." + Extension).toNullTerminatedStringRef(OutputFile);
  } else {
    if (std::error_code EC =
            sys::fs::createTemporaryFile(Prefix, Extension, OutputFile))
      return createFileError(OutputFile, EC);
  }

  TempFiles.push_back(OutputFile);
  return TempFiles.back();
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

Error runLinker(StringRef LinkerPath, ArrayRef<StringRef> LinkerArgs) {

  SmallVector<StringRef> Args({LinkerPath});
  for (StringRef Arg : LinkerArgs)
    Args.push_back(Arg);
  if (Error Err = executeCommands(LinkerPath, Args))
    return Err;
  return Error::success();
}

void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-linker-wrapper") << '\n';
}

/// Attempts to extract all the embedded device images contained inside the
/// buffer \p Contents. The buffer is expected to contain a valid offloading
/// binary format.
Error extractOffloadFiles(MemoryBufferRef Contents,
                          SmallVectorImpl<OffloadFile> &DeviceFiles) {
  uint64_t Offset = 0;
  // There could be multiple offloading binaries stored at this section.
  while (Offset < Contents.getBuffer().size()) {
    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBuffer(Contents.getBuffer().drop_front(Offset), "",
                                   /*RequiresNullTerminator*/ false);
    auto BinaryOrErr = OffloadBinary::create(*Buffer);
    if (!BinaryOrErr)
      return BinaryOrErr.takeError();
    OffloadBinary &Binary = **BinaryOrErr;

    // Create a new owned binary with a copy of the original memory.
    std::unique_ptr<MemoryBuffer> BufferCopy = MemoryBuffer::getMemBufferCopy(
        Binary.getData().take_front(Binary.getSize()),
        Contents.getBufferIdentifier());
    auto NewBinaryOrErr = OffloadBinary::create(*BufferCopy);
    if (!NewBinaryOrErr)
      return NewBinaryOrErr.takeError();
    DeviceFiles.emplace_back(std::move(*NewBinaryOrErr), std::move(BufferCopy));

    Offset += Binary.getSize();
  }

  return Error::success();
}

// Extract offloading binaries from an Object file \p Obj.
Error extractFromBinary(const ObjectFile &Obj,
                        SmallVectorImpl<OffloadFile> &DeviceFiles) {
  for (const SectionRef &Sec : Obj.sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (!Name || !Name->equals(OFFLOAD_SECTION_MAGIC_STR))
      continue;

    Expected<StringRef> Buffer = Sec.getContents();
    if (!Buffer)
      return Buffer.takeError();

    MemoryBufferRef Contents(*Buffer, Obj.getFileName());

    if (Error Err = extractOffloadFiles(Contents, DeviceFiles))
      return Err;
  }

  return Error::success();
}

Error extractFromBitcode(std::unique_ptr<MemoryBuffer> Buffer,
                         SmallVectorImpl<OffloadFile> &DeviceFiles) {
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = getLazyIRModule(std::move(Buffer), Err, Context);
  if (!M)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create module");

  // Extract offloading data from globals referenced by the
  // `llvm.embedded.object` metadata with the `.llvm.offloading` section.
  auto MD = M->getNamedMetadata("llvm.embedded.object");
  if (!MD)
    return Error::success();

  for (const MDNode *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;

    MDString *SectionID = dyn_cast<MDString>(Op->getOperand(1));
    if (!SectionID || SectionID->getString() != OFFLOAD_SECTION_MAGIC_STR)
      continue;

    GlobalVariable *GV =
        mdconst::dyn_extract_or_null<GlobalVariable>(Op->getOperand(0));
    if (!GV)
      continue;

    auto *CDS = dyn_cast<ConstantDataSequential>(GV->getInitializer());
    if (!CDS)
      continue;

    MemoryBufferRef Contents(CDS->getAsString(), M->getName());

    if (Error Err = extractOffloadFiles(Contents, DeviceFiles))
      return Err;
  }

  return Error::success();
}

Error extractFromArchive(const Archive &Library,
                         SmallVectorImpl<OffloadFile> &DeviceFiles) {
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
                        SmallVectorImpl<OffloadFile> &DeviceFiles) {
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

namespace nvptx {
Expected<StringRef> assemble(StringRef InputFile, Triple TheTriple,
                             StringRef Arch, bool RDC = true) {
  // NVPTX uses the ptxas binary to create device object files.
  Expected<std::string> PtxasPath = findProgram("ptxas", {CudaBinaryPath});
  if (!PtxasPath)
    return PtxasPath.takeError();

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                           TheTriple.getArchName() + "-" + Arch,
                       "cubin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

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
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(Opt);
  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Arch);
  if (RDC)
    CmdArgs.push_back("-c");

  CmdArgs.push_back(InputFile);

  if (Error Err = executeCommands(*PtxasPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

Expected<StringRef> link(ArrayRef<StringRef> InputFiles, Triple TheTriple,
                         StringRef Arch) {
  // NVPTX uses the nvlink binary to link device object files.
  Expected<std::string> NvlinkPath = findProgram("nvlink", {CudaBinaryPath});
  if (!NvlinkPath)
    return NvlinkPath.takeError();

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                           TheTriple.getArchName() + "-" + Arch,
                       "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*NvlinkPath);
  CmdArgs.push_back(TheTriple.isArch64Bit() ? "-m64" : "-m32");
  if (Verbose)
    CmdArgs.push_back("-v");
  if (DebugInfo != NoDebugInfo)
    CmdArgs.push_back("-g");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(Arch);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  renderXLinkerArgs(CmdArgs, TheTriple.getTriple());
  if (Error Err = executeCommands(*NvlinkPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

Expected<StringRef>
fatbinary(ArrayRef<std::pair<StringRef, StringRef>> InputFiles,
          Triple TheTriple) {
  // NVPTX uses the fatbinary program to bundle the linked images.
  Expected<std::string> FatBinaryPath =
      findProgram("fatbinary", {CudaBinaryPath});
  if (!FatBinaryPath)
    return FatBinaryPath.takeError();

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                           TheTriple.getArchName(),
                       "fatbin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*FatBinaryPath);
  CmdArgs.push_back(TheTriple.isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back("--create");
  CmdArgs.push_back(*TempFileOrErr);
  for (const auto &FileAndArch : InputFiles)
    CmdArgs.push_back(Saver.save("--image=profile=" + std::get<1>(FileAndArch) +
                                 ",file=" + std::get<0>(FileAndArch)));

  if (Error Err = executeCommands(*FatBinaryPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}
} // namespace nvptx
namespace amdgcn {
Expected<StringRef> link(ArrayRef<StringRef> InputFiles, Triple TheTriple,
                         StringRef Arch) {
  // AMDGPU uses lld to link device object files.
  Expected<std::string> LLDPath =
      findProgram("lld", {getMainExecutable("lld")});
  if (!LLDPath)
    return LLDPath.takeError();

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-" +
                           TheTriple.getArchName() + "-" + Arch,
                       "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();
  std::string ArchArg = ("-plugin-opt=mcpu=" + Arch).str();

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*LLDPath);
  CmdArgs.push_back("-flavor");
  CmdArgs.push_back("gnu");
  CmdArgs.push_back("--no-undefined");
  CmdArgs.push_back("-shared");
  CmdArgs.push_back("-plugin-opt=-amdgpu-internalize-symbols");
  CmdArgs.push_back(ArchArg);
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  renderXLinkerArgs(CmdArgs, TheTriple.getTriple());
  if (Error Err = executeCommands(*LLDPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
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

Expected<StringRef> link(ArrayRef<StringRef> InputFiles, Triple TheTriple,
                         StringRef Arch) {
  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-" +
                           TheTriple.getArchName() + "-" + Arch,
                       "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

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
  CmdArgs.push_back(*TempFileOrErr);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  renderXLinkerArgs(CmdArgs, TheTriple.getTriple());
  if (Error Err = executeCommands(LinkerUserPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}
} // namespace generic

Expected<StringRef> linkDevice(ArrayRef<StringRef> InputFiles, Triple TheTriple,
                               StringRef Arch) {
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

// Get the list of target features from the input file and unify them such that
// if there are multiple +xxx or -xxx features we only keep the last one.
std::vector<std::string> getTargetFeatures(ArrayRef<OffloadFile> InputFiles) {
  SmallVector<StringRef> Features;
  for (const OffloadFile &File : InputFiles) {
    for (auto Arg : llvm::split(File.getBinary()->getString("feature"), ","))
      Features.emplace_back(Arg);
  }

  // Only add a feature if it hasn't been seen before starting from the end.
  std::vector<std::string> UnifiedFeatures;
  DenseSet<StringRef> UsedFeatures;
  for (StringRef Feature : llvm::reverse(Features)) {
    if (UsedFeatures.insert(Feature.drop_front()).second)
      UnifiedFeatures.push_back(Feature.str());
  }

  return UnifiedFeatures;
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
    const std::vector<std::string> &Features,
    ModuleHook Hook = [](size_t, const Module &) { return true; }) {
  lto::Config Conf;
  lto::ThinBackend Backend;
  // TODO: Handle index-only thin-LTO
  Backend =
      lto::createInProcessThinBackend(llvm::heavyweight_hardware_concurrency());

  Conf.CPU = Arch.str();
  Conf.Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);

  Conf.MAttrs = Features;
  Conf.CGOptLevel = getCGOptLevel(OptLevel[1] - '0');
  Conf.OptLevel = OptLevel[1] - '0';
  if (Conf.OptLevel > 0)
    Conf.UseDefaultPipeline = true;
  Conf.DefaultTriple = TheTriple.getTriple();
  Conf.DiagHandler = diagnosticHandler;

  Conf.PTO.LoopVectorization = Conf.OptLevel > 1;
  Conf.PTO.SLPVectorization = Conf.OptLevel > 1;

  if (SaveTemps) {
    auto HandleError = [=](Error Err) {
      logAllUnhandledErrors(std::move(Err),
                            WithColor::error(errs(), LinkerExecutable));
      exit(1);
    };
    Conf.PostInternalizeModuleHook = [&, Arch](size_t, const Module &M) {
      auto TempFileOrErr =
          createOutputFile(sys::path::filename(ExecutableName) + "-" +
                               TheTriple.getTriple() + "-" + Arch,
                           "bc");
      if (!TempFileOrErr)
        HandleError(TempFileOrErr.takeError());

      std::error_code EC;
      raw_fd_ostream LinkedBitcode(*TempFileOrErr, EC, sys::fs::OF_None);
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

Error linkBitcodeFiles(SmallVectorImpl<OffloadFile> &InputFiles,
                       SmallVectorImpl<StringRef> &OutputFiles,
                       const Triple &TheTriple, StringRef Arch) {
  SmallVector<OffloadFile, 4> BitcodeInputFiles;
  DenseSet<StringRef> UsedInRegularObj;
  DenseSet<StringRef> UsedInSharedLib;

  // Search for bitcode files in the input and create an LTO input file. If it
  // is not a bitcode file, scan its symbol table for symbols we need to save.
  for (OffloadFile &File : InputFiles) {
    MemoryBufferRef Buffer = MemoryBufferRef(File.getBinary()->getImage(), "");

    file_magic Type = identify_magic(Buffer.getBuffer());
    switch (Type) {
    case file_magic::bitcode: {
      BitcodeInputFiles.emplace_back(std::move(File));
      continue;
    }
    case file_magic::cuda_fatbinary: {
      // Cuda fatbinaries made by Clang almost almost have an object eighty
      // bytes from the beginning. This should be sufficient to identify the
      // symbols.
      Buffer =
          MemoryBufferRef(Buffer.getBuffer().drop_front(FatbinaryOffset), "");
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

      for (auto &Sym : (*ObjFile)->symbols()) {
        Expected<StringRef> Name = Sym.getName();
        if (!Name)
          return Name.takeError();

        // Record if we've seen these symbols in any object or shared libraries.
        if ((*ObjFile)->isRelocatableObject())
          UsedInRegularObj.insert(*Name);
        else
          UsedInSharedLib.insert(*Name);
      }
      continue;
    }
    default:
      continue;
    }
  }

  if (BitcodeInputFiles.empty())
    return Error::success();

  // Remove all the bitcode files that we moved from the original input.
  llvm::erase_if(InputFiles, [](OffloadFile &F) { return !F.getBinary(); });

  auto HandleError = [&](Error Err) {
    logAllUnhandledErrors(std::move(Err),
                          WithColor::error(errs(), LinkerExecutable));
    exit(1);
  };

  // LTO Module hook to output bitcode without running the backend.
  SmallVector<StringRef, 4> BitcodeOutput;
  auto OutputBitcode = [&](size_t Task, const Module &M) {
    auto TempFileOrErr = createOutputFile(sys::path::filename(ExecutableName) +
                                              "-jit-" + TheTriple.getTriple(),
                                          "bc");
    if (!TempFileOrErr)
      HandleError(TempFileOrErr.takeError());

    std::error_code EC;
    raw_fd_ostream LinkedBitcode(*TempFileOrErr, EC, sys::fs::OF_None);
    if (EC)
      HandleError(errorCodeToError(EC));
    WriteBitcodeToFile(M, LinkedBitcode);
    BitcodeOutput.push_back(*TempFileOrErr);
    return false;
  };

  // We assume visibility of the whole program if every input file was bitcode.
  auto Features = getTargetFeatures(BitcodeInputFiles);
  bool WholeProgram = InputFiles.empty();
  auto LTOBackend =
      (EmbedBitcode)
          ? createLTO(TheTriple, Arch, WholeProgram, Features, OutputBitcode)
          : createLTO(TheTriple, Arch, WholeProgram, Features);

  // We need to resolve the symbols so the LTO backend knows which symbols need
  // to be kept or can be internalized. This is a simplified symbol resolution
  // scheme to approximate the full resolution a linker would do.
  DenseSet<StringRef> PrevailingSymbols;
  for (auto &BitcodeInput : BitcodeInputFiles) {
    MemoryBufferRef Buffer =
        MemoryBufferRef(BitcodeInput.getBinary()->getImage(), "");
    Expected<std::unique_ptr<lto::InputFile>> BitcodeFileOrErr =
        llvm::lto::InputFile::create(Buffer);
    if (!BitcodeFileOrErr)
      return BitcodeFileOrErr.takeError();

    // Save the input file and the buffer associated with its memory.
    const auto Symbols = (*BitcodeFileOrErr)->symbols();
    SmallVector<lto::SymbolResolution, 16> Resolutions(Symbols.size());
    size_t Idx = 0;
    for (auto &Sym : Symbols) {
      lto::SymbolResolution &Res = Resolutions[Idx++];

      // We will use this as the prevailing symbol definition in LTO unless
      // it is undefined or another definition has already been used.
      Res.Prevailing =
          !Sym.isUndefined() && PrevailingSymbols.insert(Sym.getName()).second;

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
    if (Error Err = LTOBackend->add(std::move(*BitcodeFileOrErr), Resolutions))
      return Err;
  }

  // Run the LTO job to compile the bitcode.
  size_t MaxTasks = LTOBackend->getMaxTasks();
  SmallVector<StringRef> Files(MaxTasks);
  auto AddStream = [&](size_t Task) -> std::unique_ptr<CachedFileStream> {
    int FD = -1;
    auto &TempFile = Files[Task];
    StringRef Extension = (TheTriple.isNVPTX()) ? "s" : "o";
    auto TempFileOrErr =
        createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                             TheTriple.getTriple(),
                         Extension);
    if (!TempFileOrErr)
      HandleError(TempFileOrErr.takeError());
    TempFile = *TempFileOrErr;
    if (std::error_code EC = sys::fs::openFileForWrite(TempFile, FD))
      HandleError(errorCodeToError(EC));
    return std::make_unique<CachedFileStream>(
        std::make_unique<llvm::raw_fd_ostream>(FD, true));
  };

  if (Error Err = LTOBackend->run(AddStream))
    return Err;

  // If we are embedding bitcode we only need the intermediate output.
  if (EmbedBitcode) {
    if (BitcodeOutput.size() != 1 || !WholeProgram)
      return createStringError(inconvertibleErrorCode(),
                               "Cannot embed bitcode with multiple files.");
    OutputFiles.push_back(static_cast<std::string>(BitcodeOutput.front()));
    return Error::success();
  }

  // Is we are compiling for NVPTX we need to run the assembler first.
  if (TheTriple.isNVPTX()) {
    for (StringRef &File : Files) {
      auto FileOrErr = nvptx::assemble(File, TheTriple, Arch, !WholeProgram);
      if (!FileOrErr)
        return FileOrErr.takeError();
      File = *FileOrErr;
    }
  }

  // Append the new inputs to the device linker input.
  for (StringRef File : Files)
    OutputFiles.push_back(File);

  return Error::success();
}

Expected<StringRef> writeOffloadFile(const OffloadFile &File) {
  const OffloadBinary &Binary = *File.getBinary();

  StringRef Prefix =
      sys::path::stem(Binary.getMemoryBufferRef().getBufferIdentifier());
  StringRef Suffix = getImageKindName(Binary.getImageKind());

  auto TempFileOrErr = createOutputFile(
      Prefix + "-" + Binary.getTriple() + "-" + Binary.getArch(), Suffix);
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(*TempFileOrErr, Binary.getImage().size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  std::copy(Binary.getImage().bytes_begin(), Binary.getImage().bytes_end(),
            Output->getBufferStart());
  if (Error E = Output->commit())
    return std::move(E);

  return *TempFileOrErr;
}

// Compile the module to an object file using the appropriate target machine for
// the host triple.
Expected<StringRef> compileModule(Module &M) {
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

  int FD = -1;
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-wrapper", "o");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();
  if (std::error_code EC = sys::fs::openFileForWrite(*TempFileOrErr, FD))
    return errorCodeToError(EC);

  auto OS = std::make_unique<llvm::raw_fd_ostream>(FD, true);

  legacy::PassManager CodeGenPasses;
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  CodeGenPasses.add(new TargetLibraryInfoWrapperPass(TLII));
  if (TM->addPassesToEmitFile(CodeGenPasses, *OS, nullptr, CGFT_ObjectFile))
    return createStringError(inconvertibleErrorCode(),
                             "Failed to execute host backend");
  CodeGenPasses.run(M);

  return *TempFileOrErr;
}

/// Creates the object file containing the device image and runtime
/// registration code from the device images stored in \p Images.
Expected<StringRef>
wrapDeviceImages(ArrayRef<std::unique_ptr<MemoryBuffer>> Buffers,
                 OffloadKind Kind) {
  SmallVector<ArrayRef<char>, 4> BuffersToWrap;
  for (const auto &Buffer : Buffers)
    BuffersToWrap.emplace_back(
        ArrayRef<char>(Buffer->getBufferStart(), Buffer->getBufferSize()));

  LLVMContext Context;
  Module M("offload.wrapper.module", Context);
  M.setTargetTriple(HostTriple);

  switch (Kind) {
  case OFK_OpenMP:
    if (Error Err = wrapOpenMPBinaries(M, BuffersToWrap))
      return std::move(Err);
    break;
  case OFK_Cuda:
    if (Error Err = wrapCudaBinary(M, BuffersToWrap.front()))
      return std::move(Err);
    break;
  default:
    return createStringError(inconvertibleErrorCode(),
                             getOffloadKindName(Kind) +
                                 " wrapping is not supported");
  }

  if (PrintWrappedModule)
    llvm::errs() << M;

  auto FileOrErr = compileModule(M);
  if (!FileOrErr)
    return FileOrErr.takeError();
  return *FileOrErr;
}

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleOpenMP(ArrayRef<OffloadingImage> Images) {
  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  for (const OffloadingImage &Image : Images)
    Buffers.emplace_back(
        MemoryBuffer::getMemBufferCopy(Image.Image->getBuffer()));

  return std::move(Buffers);
}

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleCuda(ArrayRef<OffloadingImage> Images) {
  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;

  SmallVector<std::pair<StringRef, StringRef>, 4> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(std::make_pair(Image.Image->getBufferIdentifier(),
                                           Image.StringData.lookup("arch")));

  Triple TheTriple = Triple(Images.front().StringData.lookup("triple"));
  auto FileOrErr = nvptx::fatbinary(InputFiles, TheTriple);
  if (!FileOrErr)
    return FileOrErr.takeError();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(*FileOrErr);
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(*FileOrErr, EC);
  Buffers.emplace_back(std::move(*ImageOrError));

  return std::move(Buffers);
}

/// Transforms the input \p Images into the binary format the runtime expects
/// for the given \p Kind.
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleLinkedOutput(ArrayRef<OffloadingImage> Images, OffloadKind Kind) {
  switch (Kind) {
  case OFK_OpenMP:
    return bundleOpenMP(Images);
  case OFK_Cuda:
    return bundleCuda(Images);
  default:
    return createStringError(inconvertibleErrorCode(),
                             getOffloadKindName(Kind) +
                                 " bundling is not supported");
  }
}

/// Transforms all the extracted offloading input files into an image that can
/// be registered by the runtime.
Expected<SmallVector<StringRef>>
linkAndWrapDeviceFiles(SmallVectorImpl<OffloadFile> &LinkerInputFiles) {
  DenseMap<OffloadFile::TargetID, SmallVector<OffloadFile, 4>> InputsForTarget;
  for (auto &File : LinkerInputFiles)
    InputsForTarget[File].emplace_back(std::move(File));
  LinkerInputFiles.clear();

  BumpPtrAllocator Alloc;
  UniqueStringSaver Saver(Alloc);
  DenseMap<OffloadKind, SmallVector<OffloadingImage, 2>> Images;
  for (auto &InputForTarget : InputsForTarget) {
    SmallVector<OffloadFile, 4> &Input = InputForTarget.getSecond();
    StringRef TripleStr = Saver.save(InputForTarget.getFirst().first);
    StringRef Arch = Saver.save(InputForTarget.getFirst().second);
    llvm::Triple Triple(TripleStr);

    DenseSet<OffloadKind> ActiveOffloadKinds;
    for (const auto &File : Input)
      ActiveOffloadKinds.insert(File.getBinary()->getOffloadKind());

    // First link and remove all the input files containing bitcode.
    SmallVector<StringRef> InputFiles;
    if (Error Err = linkBitcodeFiles(Input, InputFiles, Triple, Arch))
      return std::move(Err);

    // Write any remaining device inputs to an output file for the linker job.
    for (const OffloadFile &File : Input) {
      auto FileNameOrErr = writeOffloadFile(File);
      if (!FileNameOrErr)
        return FileNameOrErr.takeError();
      InputFiles.emplace_back(*FileNameOrErr);
    }

    // Link the remaining device files, if necessary, using the device linker.
    bool RequiresLinking =
        !Input.empty() || (!EmbedBitcode && !Triple.isNVPTX());
    auto OutputOrErr = (RequiresLinking) ? linkDevice(InputFiles, Triple, Arch)
                                         : InputFiles.front();
    if (!OutputOrErr)
      return OutputOrErr.takeError();

    // Store the offloading image for each linked output file.
    for (OffloadKind Kind : ActiveOffloadKinds) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(*OutputOrErr);
      if (std::error_code EC = FileOrErr.getError())
        return createFileError(*OutputOrErr, EC);

      OffloadingImage TheImage{};
      TheImage.TheImageKind = IMG_Object;
      TheImage.TheOffloadKind = Kind;
      TheImage.StringData = {{"triple", TripleStr}, {"arch", Arch}};
      TheImage.Image = std::move(*FileOrErr);
      Images[Kind].emplace_back(std::move(TheImage));
    }
  }

  // Create a binary image of each offloading image and embed it into a new
  // object file.
  SmallVector<StringRef> WrappedOutput;
  for (const auto &KindAndImages : Images) {
    OffloadKind Kind = KindAndImages.first;
    auto BundledImagesOrErr =
        bundleLinkedOutput(KindAndImages.second, KindAndImages.first);
    if (!BundledImagesOrErr)
      return BundledImagesOrErr.takeError();
    auto OutputOrErr = wrapDeviceImages(*BundledImagesOrErr, Kind);
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    WrappedOutput.push_back(*OutputOrErr);
  }

  return WrappedOutput;
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
  SmallVector<StringRef, 16> LinkerArgs;
  for (StringRef Arg : HostLinkerArgs)
    LinkerArgs.push_back(Arg);

  SmallVector<StringRef, 16> LibraryPaths;
  for (StringRef Arg : LinkerArgs) {
    if (Arg.startswith("-L"))
      LibraryPaths.push_back(Arg.drop_front(2));
  }

  // Try to extract device code from the linker input.
  SmallVector<OffloadFile, 4> InputFiles;
  SmallVector<OffloadFile, 4> LazyInputFiles;
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

      if (Error Err =
              extractFromBuffer(std::move(*BufferOrErr), LazyInputFiles))
        return reportError(std::move(Err));
    } else if (sys::fs::exists(Arg) && !sys::fs::is_directory(Arg)) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
          MemoryBuffer::getFileOrSTDIN(Arg);
      if (std::error_code EC = BufferOrErr.getError())
        return reportError(createFileError(Arg, EC));

      if (sys::path::extension(Arg).endswith(".a")) {
        if (Error Err =
                extractFromBuffer(std::move(*BufferOrErr), LazyInputFiles))
          return reportError(std::move(Err));
      } else {
        if (Error Err = extractFromBuffer(std::move(*BufferOrErr), InputFiles))
          return reportError(std::move(Err));
      }
    }
  }

  for (StringRef Library : BitcodeLibraries) {
    auto FileOrErr = getInputBitcodeLibrary(Library);
    if (!FileOrErr)
      return reportError(FileOrErr.takeError());
  }

  DenseSet<OffloadFile::TargetID> IsTargetUsed;
  for (const auto &File : InputFiles)
    IsTargetUsed.insert(File);

  // We should only include input files that are used.
  // TODO: Only load a library if it defined undefined symbols in the input.
  for (auto &LazyFile : LazyInputFiles)
    if (IsTargetUsed.contains(LazyFile))
      InputFiles.emplace_back(std::move(LazyFile));
  LazyInputFiles.clear();

  // Link and wrap the device images extracted from the linker input.
  auto FilesOrErr = linkAndWrapDeviceFiles(InputFiles);
  if (!FilesOrErr)
    return reportError(FilesOrErr.takeError());

  // We need to insert the new files next to the old ones to make sure they're
  // linked with the same libraries / arguments.
  if (!FilesOrErr->empty()) {
    auto *FirstInput = std::next(llvm::find_if(LinkerArgs, [](StringRef Str) {
      return sys::fs::exists(Str) && !sys::fs::is_directory(Str) &&
             Str != ExecutableName;
    }));
    LinkerArgs.insert(FirstInput, FilesOrErr->begin(), FilesOrErr->end());
  }

  // Run the host linking job.
  if (Error Err = runLinker(LinkerUserPath, LinkerArgs))
    return reportError(std::move(Err));

  // Remove the temporary files created.
  if (!SaveTemps)
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
