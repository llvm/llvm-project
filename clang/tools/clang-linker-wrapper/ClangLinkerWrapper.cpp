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
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
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
using namespace llvm::opt;
using namespace llvm::object;

/// We use the command line parser only to forward options like `-pass-remarks`
/// to the LLVM tools.
static cl::OptionCategory
    ClangLinkerWrapperCategory("clang-linker-wrapper options");
static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden,
                          cl::cat(ClangLinkerWrapperCategory));
static cl::list<std::string>
    DummyArguments(cl::Sink, cl::Hidden, cl::cat(ClangLinkerWrapperCategory));

/// Path of the current binary.
static const char *LinkerExecutable;

/// Ssave intermediary results.
static bool SaveTemps = false;

/// Print arguments without executing.
static bool DryRun = false;

/// Print verbose output.
static bool Verbose = false;

/// Filename of the executable being created.
static StringRef ExecutableName;

/// Binary path for the CUDA installation.
static std::string CudaBinaryPath;

/// Temporary files created by the linker wrapper.
static std::list<SmallString<128>> TempFiles;

/// Codegen flags for LTO backend.
static codegen::RegisterCodeGenFlags CodeGenFlags;

/// Global flag to indicate that the LTO pipeline threw an error.
static std::atomic<bool> LTOError;

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
using std::error_code;

/// Must not overlap with llvm::opt::DriverFlag.
enum WrapperFlags {
  WrapperOnlyOption = (1 << 4), // Options only used by the linker wrapper.
  DeviceOnlyOption = (1 << 5),  // Options only used for device linking.
};

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "LinkerWrapperOpts.inc"
  LastOption
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "LinkerWrapperOpts.inc"
#undef PREFIX

static const OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {PREFIX, NAME,  HELPTEXT,    METAVAR,     OPT_##ID,  Option::KIND##Class,    \
   PARAM,  FLAGS, OPT_##GROUP, OPT_##ALIAS, ALIASARGS, VALUES},
#include "LinkerWrapperOpts.inc"
#undef OPTION
};

class WrapperOptTable : public opt::OptTable {
public:
  WrapperOptTable() : OptTable(InfoTable) {}
};

const OptTable &getOptTable() {
  static const WrapperOptTable *Table = []() {
    auto Result = std::make_unique<WrapperOptTable>();
    return Result.release();
  }();
  return *Table;
}

Error extractFromBuffer(std::unique_ptr<MemoryBuffer> Buffer,
                        SmallVectorImpl<OffloadFile> &DeviceFiles);

void printCommands(ArrayRef<StringRef> CmdArgs) {
  if (CmdArgs.empty())
    return;

  llvm::errs() << " \"" << CmdArgs.front() << "\" ";
  for (auto IC = std::next(CmdArgs.begin()), IE = CmdArgs.end(); IC != IE; ++IC)
    llvm::errs() << *IC << (std::next(IC) != IE ? " " : "\n");
}

[[noreturn]] void reportError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E),
                        WithColor::error(errs(), LinkerExecutable));
  exit(EXIT_FAILURE);
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

  TempFiles.emplace_back(std::move(OutputFile));
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

/// Runs the wrapped linker job with the newly created input.
Error runLinker(ArrayRef<StringRef> Files, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("Execute host linker");

  // Render the linker arguments and add the newly created image. We add it
  // after the output file to ensure it is linked with the correct libraries.
  StringRef LinkerPath = Args.getLastArgValue(OPT_linker_path_EQ);
  ArgStringList NewLinkerArgs;
  for (const opt::Arg *Arg : Args) {
    // Do not forward arguments only intended for the linker wrapper.
    if (Arg->getOption().hasFlag(WrapperOnlyOption))
      continue;

    Arg->render(Args, NewLinkerArgs);
    if (Arg->getOption().matches(OPT_o))
      llvm::transform(Files, std::back_inserter(NewLinkerArgs),
                      [&](StringRef Arg) { return Args.MakeArgString(Arg); });
  }

  SmallVector<StringRef> LinkerArgs({LinkerPath});
  for (StringRef Arg : NewLinkerArgs)
    LinkerArgs.push_back(Arg);
  if (Error Err = executeCommands(LinkerPath, LinkerArgs))
    return Err;
  return Error::success();
}

void printVersion(raw_ostream &OS) {
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
  for (ELFSectionRef Sec : Obj.sections()) {
    if (Sec.getType() != ELF::SHT_LLVM_OFFLOADING)
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
  auto *MD = M->getNamedMetadata("llvm.embedded.objects");
  if (!MD)
    return Error::success();

  for (const MDNode *Op : MD->operands()) {
    if (Op->getNumOperands() < 2)
      continue;

    MDString *SectionID = dyn_cast<MDString>(Op->getOperand(1));
    if (!SectionID || SectionID->getString() != ".llvm.offloading")
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
  case file_magic::elf_relocatable: {
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
Expected<StringRef> assemble(StringRef InputFile, const ArgList &Args,
                             bool RDC = true) {
  llvm::TimeTraceScope TimeScope("NVPTX Assembler");
  // NVPTX uses the ptxas binary to create device object files.
  Expected<std::string> PtxasPath = findProgram("ptxas", {CudaBinaryPath});
  if (!PtxasPath)
    return PtxasPath.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);
  // Create a new file to write the linked device image to. Assume that the
  // input filename already has the device and architecture.
  auto TempFileOrErr = createOutputFile(sys::path::stem(InputFile), "cubin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs;
  StringRef OptLevel = Args.getLastArgValue(OPT_opt_level, "O2");
  CmdArgs.push_back(*PtxasPath);
  CmdArgs.push_back(Triple.isArch64Bit() ? "-m64" : "-m32");
  if (Verbose)
    CmdArgs.push_back("-v");
  for (StringRef Arg : Args.getAllArgValues(OPT_ptxas_arg))
    CmdArgs.push_back(Args.MakeArgString(Arg));
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back(Args.MakeArgString("-" + OptLevel));
  CmdArgs.push_back("--gpu-name");
  CmdArgs.push_back(Arch);
  if (Args.hasArg(OPT_debug))
    CmdArgs.push_back("-g");
  if (RDC)
    CmdArgs.push_back("-c");

  CmdArgs.push_back(InputFile);

  if (Error Err = executeCommands(*PtxasPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

Expected<StringRef> link(ArrayRef<StringRef> InputFiles, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("NVPTX linker");
  // NVPTX uses the nvlink binary to link device object files.
  Expected<std::string> NvlinkPath = findProgram("nvlink", {CudaBinaryPath});
  if (!NvlinkPath)
    return NvlinkPath.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                           Triple.getArchName() + "-" + Arch,
                       "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*NvlinkPath);
  CmdArgs.push_back(Triple.isArch64Bit() ? "-m64" : "-m32");
  if (Args.hasArg(OPT_debug))
    CmdArgs.push_back("-g");
  if (Verbose)
    CmdArgs.push_back("-v");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(Arch);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  for (StringRef Arg : Args.getAllArgValues(OPT_linker_arg_EQ))
    CmdArgs.push_back(Args.MakeArgString(Arg));
  if (Error Err = executeCommands(*NvlinkPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

Expected<StringRef>
fatbinary(ArrayRef<std::pair<StringRef, StringRef>> InputFiles,
          const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("NVPTX fatbinary");
  // NVPTX uses the fatbinary program to bundle the linked images.
  Expected<std::string> FatBinaryPath =
      findProgram("fatbinary", {CudaBinaryPath});
  if (!FatBinaryPath)
    return FatBinaryPath.takeError();

  llvm::Triple Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));

  // Create a new file to write the linked device image to.
  auto TempFileOrErr = createOutputFile(
      sys::path::filename(ExecutableName) + "-device", "fatbin");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*FatBinaryPath);
  CmdArgs.push_back(Triple.isArch64Bit() ? "-64" : "-32");
  CmdArgs.push_back("--create");
  CmdArgs.push_back(*TempFileOrErr);
  for (const auto &FileAndArch : InputFiles)
    CmdArgs.push_back(
        Args.MakeArgString("--image=profile=" + std::get<1>(FileAndArch) +
                           ",file=" + std::get<0>(FileAndArch)));

  if (Error Err = executeCommands(*FatBinaryPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}
} // namespace nvptx

namespace amdgcn {
Expected<StringRef> link(ArrayRef<StringRef> InputFiles, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("AMDGPU linker");
  // AMDGPU uses lld to link device object files.
  Expected<std::string> LLDPath =
      findProgram("lld", {getMainExecutable("lld")});
  if (!LLDPath)
    return LLDPath.takeError();

  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-" +
                           Triple.getArchName() + "-" + Arch,
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

  for (StringRef Arg : Args.getAllArgValues(OPT_linker_arg_EQ))
    CmdArgs.push_back(Args.MakeArgString(Arg));
  if (Error Err = executeCommands(*LLDPath, CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}

Expected<StringRef>
fatbinary(ArrayRef<std::pair<StringRef, StringRef>> InputFiles,
          const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("AMDGPU Fatbinary");

  // AMDGPU uses the clang-offload-bundler to bundle the linked images.
  Expected<std::string> OffloadBundlerPath = findProgram(
      "clang-offload-bundler", {getMainExecutable("clang-offload-bundler")});
  if (!OffloadBundlerPath)
    return OffloadBundlerPath.takeError();

  llvm::Triple Triple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));

  // Create a new file to write the linked device image to.
  auto TempFileOrErr = createOutputFile(sys::path::filename(ExecutableName) +
                                            "-device-" + Triple.getArchName(),
                                        "hipfb");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*OffloadBundlerPath);
  CmdArgs.push_back("-type=o");
  CmdArgs.push_back("-bundle-align=4096");

  SmallVector<StringRef> Targets = {"-targets=host-x86_64-unknown-linux"};
  for (const auto &FileAndArch : InputFiles)
    Targets.push_back(
        Saver.save("hipv4-amdgcn-amd-amdhsa--" + std::get<1>(FileAndArch)));
  CmdArgs.push_back(Saver.save(llvm::join(Targets, ",")));

  CmdArgs.push_back("-input=/dev/null");
  for (const auto &FileAndArch : InputFiles)
    CmdArgs.push_back(Saver.save("-input=" + std::get<0>(FileAndArch)));

  CmdArgs.push_back(Saver.save("-output=" + *TempFileOrErr));

  if (Error Err = executeCommands(*OffloadBundlerPath, CmdArgs))
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

Expected<StringRef> link(ArrayRef<StringRef> InputFiles, const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("Generic linker");
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);

  // Create a new file to write the linked device image to.
  auto TempFileOrErr =
      createOutputFile(sys::path::filename(ExecutableName) + "-" +
                           Triple.getArchName() + "-" + Arch,
                       "out");
  if (!TempFileOrErr)
    return TempFileOrErr.takeError();

  // Use the host linker to perform generic offloading. Use the same libraries
  // and paths as the host application does.
  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(Args.getLastArgValue(OPT_linker_path_EQ));
  CmdArgs.push_back("-m");
  CmdArgs.push_back(getLDMOption(Triple));
  CmdArgs.push_back("-shared");

  ArgStringList LinkerArgs;
  for (const opt::Arg *Arg : Args) {
    auto Op = Arg->getOption();
    if (Op.matches(OPT_library) || Op.matches(OPT_library_path) ||
        Op.matches(OPT_as_needed) || Op.matches(OPT_no_as_needed) ||
        Op.matches(OPT_rpath) || Op.matches(OPT_dynamic_linker))
      Arg->render(Args, LinkerArgs);
  }
  for (StringRef Arg : LinkerArgs)
    CmdArgs.push_back(Arg);

  CmdArgs.push_back("-Bsymbolic");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(*TempFileOrErr);

  // Add extracted input files.
  for (StringRef Input : InputFiles)
    CmdArgs.push_back(Input);

  for (StringRef Arg : Args.getAllArgValues(OPT_linker_arg_EQ))
    CmdArgs.push_back(Args.MakeArgString(Arg));
  if (Error Err =
          executeCommands(Args.getLastArgValue(OPT_linker_path_EQ), CmdArgs))
    return std::move(Err);

  return *TempFileOrErr;
}
} // namespace generic

Expected<StringRef> linkDevice(ArrayRef<StringRef> InputFiles,
                               const ArgList &Args) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  switch (Triple.getArch()) {
  case Triple::nvptx:
  case Triple::nvptx64:
    return nvptx::link(InputFiles, Args);
  case Triple::amdgcn:
    return amdgcn::link(InputFiles, Args);
  case Triple::x86:
  case Triple::x86_64:
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::ppc64:
  case Triple::ppc64le:
    return generic::link(InputFiles, Args);
  default:
    return createStringError(inconvertibleErrorCode(),
                             Triple.getArchName() +
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
    LTOError = true;
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
    const ArgList &Args, const std::vector<std::string> &Features,
    ModuleHook Hook = [](size_t, const Module &) { return true; }) {
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));
  StringRef Arch = Args.getLastArgValue(OPT_arch_EQ);
  lto::Config Conf;
  lto::ThinBackend Backend;
  // TODO: Handle index-only thin-LTO
  Backend =
      lto::createInProcessThinBackend(llvm::heavyweight_hardware_concurrency());

  Conf.CPU = Arch.str();
  Conf.Options = codegen::InitTargetOptionsFromCodeGenFlags(Triple);

  StringRef OptLevel = Args.getLastArgValue(OPT_opt_level, "O2");
  Conf.MAttrs = Features;
  Conf.CGOptLevel = getCGOptLevel(OptLevel[1] - '0');
  Conf.OptLevel = OptLevel[1] - '0';
  if (Conf.OptLevel > 0)
    Conf.UseDefaultPipeline = true;
  Conf.DefaultTriple = Triple.getTriple();

  LTOError = false;
  Conf.DiagHandler = diagnosticHandler;

  Conf.PTO.LoopVectorization = Conf.OptLevel > 1;
  Conf.PTO.SLPVectorization = Conf.OptLevel > 1;

  if (SaveTemps) {
    std::string TempName = (sys::path::filename(ExecutableName) + "-device-" +
                            Triple.getTriple() + "-" + Arch)
                               .str();
    Conf.PostInternalizeModuleHook = [=](size_t Task, const Module &M) {
      std::string File = !Task ? TempName + ".bc"
                               : TempName + "." + std::to_string(Task) + ".bc";
      error_code EC;
      raw_fd_ostream LinkedBitcode(File, EC, sys::fs::OF_None);
      if (EC)
        reportError(errorCodeToError(EC));
      WriteBitcodeToFile(M, LinkedBitcode);
      return true;
    };
  }
  Conf.PostOptModuleHook = Hook;
  Conf.CGFileType = Triple.isNVPTX() ? CGFT_AssemblyFile : CGFT_ObjectFile;

  // TODO: Handle remark files
  Conf.HasWholeProgramVisibility = Args.hasArg(OPT_whole_program);

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
                       const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("Link bitcode files");
  const llvm::Triple Triple(Args.getLastArgValue(OPT_triple_EQ));

  SmallVector<OffloadFile, 4> BitcodeInputFiles;
  DenseSet<StringRef> UsedInRegularObj;
  DenseSet<StringRef> UsedInSharedLib;
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

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
    case file_magic::elf_relocatable:
    case file_magic::elf_shared_object: {
      Expected<std::unique_ptr<ObjectFile>> ObjFile =
          ObjectFile::createObjectFile(Buffer);
      if (!ObjFile)
        continue;

      for (SymbolRef Sym : (*ObjFile)->symbols()) {
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

  if (BitcodeInputFiles.empty())
    return Error::success();

  // Remove all the bitcode files that we moved from the original input.
  llvm::erase_if(InputFiles, [](OffloadFile &F) { return !F.getBinary(); });

  // LTO Module hook to output bitcode without running the backend.
  SmallVector<StringRef, 4> BitcodeOutput;
  auto OutputBitcode = [&](size_t, const Module &M) {
    auto TempFileOrErr = createOutputFile(sys::path::filename(ExecutableName) +
                                              "-jit-" + Triple.getTriple(),
                                          "bc");
    if (!TempFileOrErr)
      reportError(TempFileOrErr.takeError());

    std::error_code EC;
    raw_fd_ostream LinkedBitcode(*TempFileOrErr, EC, sys::fs::OF_None);
    if (EC)
      reportError(errorCodeToError(EC));
    WriteBitcodeToFile(M, LinkedBitcode);
    BitcodeOutput.push_back(*TempFileOrErr);
    return false;
  };

  // We assume visibility of the whole program if every input file was bitcode.
  auto Features = getTargetFeatures(BitcodeInputFiles);
  auto LTOBackend = Args.hasArg(OPT_embed_bitcode)
                        ? createLTO(Args, Features, OutputBitcode)
                        : createLTO(Args, Features);

  // We need to resolve the symbols so the LTO backend knows which symbols need
  // to be kept or can be internalized. This is a simplified symbol resolution
  // scheme to approximate the full resolution a linker would do.
  uint64_t Idx = 0;
  DenseSet<StringRef> PrevailingSymbols;
  for (auto &BitcodeInput : BitcodeInputFiles) {
    // Get a semi-unique buffer identifier for Thin-LTO.
    StringRef Identifier = Saver.save(
        std::to_string(Idx++) + "." +
        BitcodeInput.getBinary()->getMemoryBufferRef().getBufferIdentifier());
    MemoryBufferRef Buffer =
        MemoryBufferRef(BitcodeInput.getBinary()->getImage(), Identifier);
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
    if (Error Err = LTOBackend->add(std::move(*BitcodeFileOrErr), Resolutions))
      return Err;
  }

  // Run the LTO job to compile the bitcode.
  size_t MaxTasks = LTOBackend->getMaxTasks();
  SmallVector<StringRef> Files(MaxTasks);
  auto AddStream = [&](size_t Task) -> std::unique_ptr<CachedFileStream> {
    int FD = -1;
    auto &TempFile = Files[Task];
    StringRef Extension = (Triple.isNVPTX()) ? "s" : "o";
    std::string TaskStr = Task ? "." + std::to_string(Task) : "";
    auto TempFileOrErr =
        createOutputFile(sys::path::filename(ExecutableName) + "-device-" +
                             Triple.getTriple() + TaskStr,
                         Extension);
    if (!TempFileOrErr)
      reportError(TempFileOrErr.takeError());
    TempFile = *TempFileOrErr;
    if (std::error_code EC = sys::fs::openFileForWrite(TempFile, FD))
      reportError(errorCodeToError(EC));
    return std::make_unique<CachedFileStream>(
        std::make_unique<llvm::raw_fd_ostream>(FD, true));
  };

  if (Error Err = LTOBackend->run(AddStream))
    return Err;

  if (LTOError)
    return createStringError(inconvertibleErrorCode(),
                             "Errors encountered inside the LTO pipeline.");

  // If we are embedding bitcode we only need the intermediate output.
  bool SingleOutput = Files.size() == 1;
  if (Args.hasArg(OPT_embed_bitcode)) {
    if (BitcodeOutput.size() != 1 || !SingleOutput)
      return createStringError(inconvertibleErrorCode(),
                               "Cannot embed bitcode with multiple files.");
    OutputFiles.push_back(static_cast<std::string>(BitcodeOutput.front()));
    return Error::success();
  }

  // Is we are compiling for NVPTX we need to run the assembler first.
  if (Triple.isNVPTX()) {
    for (StringRef &File : Files) {
      auto FileOrErr = nvptx::assemble(File, Args, !SingleOutput);
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
  llvm::TimeTraceScope TimeScope("Compile module");
  std::string Msg;
  const Target *T = TargetRegistry::lookupTarget(M.getTargetTriple(), Msg);
  if (!T)
    return createStringError(inconvertibleErrorCode(), Msg);

  auto Options =
      codegen::InitTargetOptionsFromCodeGenFlags(Triple(M.getTargetTriple()));
  StringRef CPU = "";
  StringRef Features = "";
  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(M.getTargetTriple(), CPU, Features, Options,
                             Reloc::PIC_, M.getCodeModel()));

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
                 const ArgList &Args, OffloadKind Kind) {
  llvm::TimeTraceScope TimeScope("Wrap bundled images");

  SmallVector<ArrayRef<char>, 4> BuffersToWrap;
  for (const auto &Buffer : Buffers)
    BuffersToWrap.emplace_back(
        ArrayRef<char>(Buffer->getBufferStart(), Buffer->getBufferSize()));

  LLVMContext Context;
  Module M("offload.wrapper.module", Context);
  M.setTargetTriple(
      Args.getLastArgValue(OPT_host_triple_EQ, sys::getDefaultTargetTriple()));

  switch (Kind) {
  case OFK_OpenMP:
    if (Error Err = wrapOpenMPBinaries(M, BuffersToWrap))
      return std::move(Err);
    break;
  case OFK_Cuda:
    if (Error Err = wrapCudaBinary(M, BuffersToWrap.front()))
      return std::move(Err);
    break;
  case OFK_HIP:
    if (Error Err = wrapHIPBinary(M, BuffersToWrap.front()))
      return std::move(Err);
    break;
  default:
    return createStringError(inconvertibleErrorCode(),
                             getOffloadKindName(Kind) +
                                 " wrapping is not supported");
  }

  if (Args.hasArg(OPT_print_wrapped_module))
    errs() << M;

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
bundleCuda(ArrayRef<OffloadingImage> Images, const ArgList &Args) {
  SmallVector<std::pair<StringRef, StringRef>, 4> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(std::make_pair(Image.Image->getBufferIdentifier(),
                                           Image.StringData.lookup("arch")));

  Triple TheTriple = Triple(Images.front().StringData.lookup("triple"));
  auto FileOrErr = nvptx::fatbinary(InputFiles, Args);
  if (!FileOrErr)
    return FileOrErr.takeError();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(*FileOrErr);

  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(*FileOrErr, EC);
  Buffers.emplace_back(std::move(*ImageOrError));

  return std::move(Buffers);
}

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleHIP(ArrayRef<OffloadingImage> Images, const ArgList &Args) {
  SmallVector<std::pair<StringRef, StringRef>, 4> InputFiles;
  for (const OffloadingImage &Image : Images)
    InputFiles.emplace_back(std::make_pair(Image.Image->getBufferIdentifier(),
                                           Image.StringData.lookup("arch")));

  Triple TheTriple = Triple(Images.front().StringData.lookup("triple"));
  auto FileOrErr = amdgcn::fatbinary(InputFiles, Args);
  if (!FileOrErr)
    return FileOrErr.takeError();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(*FileOrErr);

  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  if (std::error_code EC = ImageOrError.getError())
    return createFileError(*FileOrErr, EC);
  Buffers.emplace_back(std::move(*ImageOrError));

  return std::move(Buffers);
}

/// Transforms the input \p Images into the binary format the runtime expects
/// for the given \p Kind.
Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleLinkedOutput(ArrayRef<OffloadingImage> Images, const ArgList &Args,
                   OffloadKind Kind) {
  llvm::TimeTraceScope TimeScope("Bundle linked output");
  switch (Kind) {
  case OFK_OpenMP:
    return bundleOpenMP(Images);
  case OFK_Cuda:
    return bundleCuda(Images, Args);
  case OFK_HIP:
    return bundleHIP(Images, Args);
  default:
    return createStringError(inconvertibleErrorCode(),
                             getOffloadKindName(Kind) +
                                 " bundling is not supported");
  }
}

/// Returns a new ArgList containg arguments used for the device linking phase.
DerivedArgList getLinkerArgs(ArrayRef<OffloadFile> Input,
                             const InputArgList &Args) {
  DerivedArgList DAL = DerivedArgList(DerivedArgList(Args));
  for (Arg *A : Args)
    DAL.append(A);

  // Set the subarchitecture and target triple for this compilation.
  const OptTable &Tbl = getOptTable();
  DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_arch_EQ),
                   Args.MakeArgString(Input.front().getBinary()->getArch()));
  DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_triple_EQ),
                   Args.MakeArgString(Input.front().getBinary()->getTriple()));

  // If every input file is bitcode we have whole program visibility as we do
  // only support static linking with bitcode.
  auto ContainsBitcode = [](const OffloadFile &F) {
    return identify_magic(F.getBinary()->getImage()) == file_magic::bitcode;
  };
  if (llvm::all_of(Input, ContainsBitcode))
    DAL.AddFlagArg(nullptr, Tbl.getOption(OPT_whole_program));

  // Forward '-Xoffload-linker' options to the appropriate backend.
  for (StringRef Arg : Args.getAllArgValues(OPT_device_linker_args_EQ)) {
    auto TripleAndValue = Arg.split('=');
    if (TripleAndValue.second.empty())
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_linker_arg_EQ),
                       Args.MakeArgString(TripleAndValue.first));
    else if (TripleAndValue.first == DAL.getLastArgValue(OPT_triple_EQ))
      DAL.AddJoinedArg(nullptr, Tbl.getOption(OPT_linker_arg_EQ),
                       Args.MakeArgString(TripleAndValue.second));
  }

  return DAL;
}

/// Transforms all the extracted offloading input files into an image that can
/// be registered by the runtime.
Expected<SmallVector<StringRef>>
linkAndWrapDeviceFiles(SmallVectorImpl<OffloadFile> &LinkerInputFiles,
                       const InputArgList &Args) {
  llvm::TimeTraceScope TimeScope("Handle all device input");

  DenseMap<OffloadFile::TargetID, SmallVector<OffloadFile, 4>> InputsForTarget;
  for (auto &File : LinkerInputFiles)
    InputsForTarget[File].emplace_back(std::move(File));
  LinkerInputFiles.clear();

  DenseMap<OffloadKind, SmallVector<OffloadingImage, 2>> Images;
  for (auto &InputForTarget : InputsForTarget) {
    llvm::TimeTraceScope TimeScope("Link device input");

    SmallVector<OffloadFile, 4> &Input = InputForTarget.getSecond();
    auto LinkerArgs = getLinkerArgs(Input, Args);

    DenseSet<OffloadKind> ActiveOffloadKinds;
    for (const auto &File : Input)
      ActiveOffloadKinds.insert(File.getBinary()->getOffloadKind());

    // First link and remove all the input files containing bitcode.
    SmallVector<StringRef> InputFiles;
    if (Error Err = linkBitcodeFiles(Input, InputFiles, LinkerArgs))
      return std::move(Err);

    // Write any remaining device inputs to an output file for the linker job.
    for (const OffloadFile &File : Input) {
      auto FileNameOrErr = writeOffloadFile(File);
      if (!FileNameOrErr)
        return FileNameOrErr.takeError();
      InputFiles.emplace_back(*FileNameOrErr);
    }

    // Link the remaining device files, if necessary, using the device linker.
    llvm::Triple Triple(LinkerArgs.getLastArgValue(OPT_triple_EQ));
    bool RequiresLinking =
        !Args.hasArg(OPT_embed_bitcode) &&
        !(Input.empty() && InputFiles.size() == 1 && Triple.isNVPTX());
    auto OutputOrErr = RequiresLinking ? linkDevice(InputFiles, LinkerArgs)
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
      TheImage.StringData = {
          {"triple", LinkerArgs.getLastArgValue(OPT_triple_EQ)},
          {"arch", LinkerArgs.getLastArgValue(OPT_arch_EQ)}};
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
        bundleLinkedOutput(KindAndImages.second, Args, Kind);
    if (!BundledImagesOrErr)
      return BundledImagesOrErr.takeError();
    auto OutputOrErr = wrapDeviceImages(*BundledImagesOrErr, Args, Kind);
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    WrappedOutput.push_back(*OutputOrErr);
  }

  return WrappedOutput;
}

Optional<std::string> findFile(StringRef Dir, StringRef Root,
                               const Twine &Name) {
  SmallString<128> Path;
  if (Dir.startswith("="))
    sys::path::append(Path, Root, Dir.substr(1), Name);
  else
    sys::path::append(Path, Dir, Name);

  if (sys::fs::exists(Path))
    return static_cast<std::string>(Path);
  return None;
}

Optional<std::string> findFromSearchPaths(StringRef Name, StringRef Root,
                                          ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths)
    if (Optional<std::string> File = findFile(Dir, Root, Name))
      return File;
  return None;
}

Optional<std::string> searchLibraryBaseName(StringRef Name, StringRef Root,
                                            ArrayRef<StringRef> SearchPaths) {
  for (StringRef Dir : SearchPaths) {
    if (Optional<std::string> File = findFile(Dir, Root, "lib" + Name + ".so"))
      return None;
    if (Optional<std::string> File = findFile(Dir, Root, "lib" + Name + ".a"))
      return File;
  }
  return None;
}

/// Search for static libraries in the linker's library path given input like
/// `-lfoo` or `-l:libfoo.a`.
Optional<std::string> searchLibrary(StringRef Input, StringRef Root,
                                    ArrayRef<StringRef> SearchPaths) {
  if (Input.startswith(":"))
    return findFromSearchPaths(Input.drop_front(), Root, SearchPaths);
  return searchLibraryBaseName(Input, Root, SearchPaths);
}

/// Search the input files and libraries for embedded device offloading code and
/// add it to the list of files to be linked. Files coming from static libraries
/// are only added to the input if they are used by an existing input file.
Expected<SmallVector<OffloadFile>> getDeviceInput(const ArgList &Args) {
  llvm::TimeTraceScope TimeScope("ExtractDeviceCode");

  StringRef Root = Args.getLastArgValue(OPT_sysroot_EQ);
  SmallVector<StringRef> LibraryPaths;
  for (const opt::Arg *Arg : Args.filtered(OPT_library_path))
    LibraryPaths.push_back(Arg->getValue());

  // Try to extract device code from the linker input files.
  SmallVector<OffloadFile> InputFiles;
  SmallVector<OffloadFile> LazyInputFiles;
  for (const opt::Arg *Arg : Args.filtered(OPT_INPUT)) {
    StringRef Filename = Arg->getValue();
    if (!sys::fs::exists(Filename) || sys::fs::is_directory(Filename))
      continue;

    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(Filename);
    if (std::error_code EC = BufferOrErr.getError())
      reportError(createFileError(Filename, EC));

    bool IsLazy =
        identify_magic((*BufferOrErr)->getBuffer()) == file_magic::archive;
    if (Error Err = extractFromBuffer(std::move(*BufferOrErr),
                                      IsLazy ? LazyInputFiles : InputFiles))
      reportError(std::move(Err));
  }

  // Try to extract input from input libraries.
  for (const opt::Arg *Arg : Args.filtered(OPT_library)) {
    if (auto Library = searchLibrary(Arg->getValue(), Root, LibraryPaths)) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
          MemoryBuffer::getFileOrSTDIN(*Library);
      if (std::error_code EC = BufferOrErr.getError())
        reportError(createFileError(*Library, EC));

      if (Error Err =
              extractFromBuffer(std::move(*BufferOrErr), LazyInputFiles))
        reportError(std::move(Err));
    }
  }

  for (StringRef Library : Args.getAllArgValues(OPT_bitcode_library_EQ)) {
    auto FileOrErr = getInputBitcodeLibrary(Library);
    if (!FileOrErr)
      reportError(FileOrErr.takeError());
    InputFiles.push_back(std::move(*FileOrErr));
  }

  DenseSet<OffloadFile::TargetID> IsTargetUsed;
  for (const auto &File : InputFiles)
    IsTargetUsed.insert(File);

  // We should only include input files that are used.
  // TODO: Only load a library if it defined undefined symbols in the input.
  for (auto &LazyFile : LazyInputFiles)
    if (IsTargetUsed.contains(LazyFile))
      InputFiles.emplace_back(std::move(LazyFile));

  return InputFiles;
}

} // namespace

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmParsers();
  InitializeAllAsmPrinters();

  LinkerExecutable = Argv[0];
  sys::PrintStackTraceOnErrorSignal(Argv[0]);

  const OptTable &Tbl = getOptTable();
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  auto Args = Tbl.parseArgs(Argc, Argv, OPT_INVALID, Saver, [&](StringRef Err) {
    reportError(createStringError(inconvertibleErrorCode(), Err));
  });

  if (Args.hasArg(OPT_help) || Args.hasArg(OPT_help_hidden)) {
    Tbl.printHelp(
        outs(),
        "clang-linker-wrapper [options] -- <options to passed to the linker>",
        "\nA wrapper utility over the host linker. It scans the input files\n"
        "for sections that require additional processing prior to linking.\n"
        "The will then transparently pass all arguments and input to the\n"
        "specified host linker to create the final binary.\n",
        Args.hasArg(OPT_help_hidden), Args.hasArg(OPT_help_hidden));
    return EXIT_SUCCESS;
  }
  if (Args.hasArg(OPT_v)) {
    printVersion(outs());
    return EXIT_SUCCESS;
  }

  // This forwards '-pass-remarks=' to the LTO backend if present.
  cl::HideUnrelatedOptions(ClangLinkerWrapperCategory);
  cl::ParseCommandLineOptions(Argc, Argv);

  Verbose = Args.hasArg(OPT_verbose);
  DryRun = Args.hasArg(OPT_dry_run);
  SaveTemps = Args.hasArg(OPT_save_temps);
  ExecutableName = Args.getLastArgValue(OPT_o, "a.out");
  CudaBinaryPath = Args.getLastArgValue(OPT_cuda_path_EQ).str();
  if (!CudaBinaryPath.empty())
    CudaBinaryPath = CudaBinaryPath + "/bin";

  if (Args.hasArg(OPT_wrapper_time_trace_eq)) {
    unsigned Granularity;
    Args.getLastArgValue(OPT_wrapper_time_trace_granularity, "500")
        .getAsInteger(10, Granularity);
    timeTraceProfilerInitialize(Granularity, Argv[0]);
  }

  {
    llvm::TimeTraceScope TimeScope("Execute linker wrapper");

    // Extract the device input files stored in the host fat binary.
    auto DeviceInputFiles = getDeviceInput(Args);
    if (!DeviceInputFiles)
      reportError(DeviceInputFiles.takeError());

    // Link and wrap the device images extracted from the linker input.
    auto FilesOrErr = linkAndWrapDeviceFiles(*DeviceInputFiles, Args);
    if (!FilesOrErr)
      reportError(FilesOrErr.takeError());

    // Run the host linking job with the rendered arguments.
    if (Error Err = runLinker(*FilesOrErr, Args))
      reportError(std::move(Err));
  }

  if (const opt::Arg *Arg = Args.getLastArg(OPT_wrapper_time_trace_eq)) {
    if (Error Err = timeTraceProfilerWrite(Arg->getValue(), ExecutableName))
      reportError(std::move(Err));
    timeTraceProfilerCleanup();
  }

  // Remove the temporary files created.
  if (!SaveTemps)
    for (const auto &TempFile : TempFiles)
      if (std::error_code EC = sys::fs::remove(TempFile))
        reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
