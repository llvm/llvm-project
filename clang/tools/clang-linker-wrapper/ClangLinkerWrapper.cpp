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

static cl::opt<bool> StripSections(
    "strip-sections", cl::ZeroOrMore,
    cl::desc("Strip offloading sections from the host object file."),
    cl::init(true), cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string> LinkerUserPath("linker-path", cl::Required,
                                           cl::desc("Path of linker binary"),
                                           cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string>
    TargetFeatures("target-feature", cl::ZeroOrMore,
                   cl::desc("Target features for triple"),
                   cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string> OptLevel("opt-level", cl::ZeroOrMore,
                                     cl::desc("Optimization level for LTO"),
                                     cl::init("O2"),
                                     cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    BitcodeLibraries("target-library", cl::ZeroOrMore,
                     cl::desc("Path for the target bitcode library"),
                     cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool> EmbedBitcode(
    "target-embed-bc", cl::ZeroOrMore,
    cl::desc("Embed linked bitcode instead of an executable device image"),
    cl::init(false), cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string>
    HostTriple("host-triple", cl::ZeroOrMore,
               cl::desc("Triple to use for the host compilation"),
               cl::init(sys::getDefaultTargetTriple()),
               cl::cat(ClangLinkerWrapperCategory));

static cl::list<std::string>
    PtxasArgs("ptxas-args", cl::ZeroOrMore,
              cl::desc("Argument to pass to the ptxas invocation"),
              cl::cat(ClangLinkerWrapperCategory));

static cl::opt<bool> Verbose("v", cl::ZeroOrMore,
                             cl::desc("Verbose output from tools"),
                             cl::init(false),
                             cl::cat(ClangLinkerWrapperCategory));

static cl::opt<DebugKind> DebugInfo(
    cl::desc("Choose debugging level:"), cl::init(NoDebugInfo),
    cl::values(clEnumValN(NoDebugInfo, "g0", "No debug information"),
               clEnumValN(DirectivesOnly, "gline-directives-only",
                          "Direction information"),
               clEnumValN(FullDebugInfo, "g", "Full debugging support")));

static cl::opt<bool> SaveTemps("save-temps", cl::ZeroOrMore,
                               cl::desc("Save intermediary results."),
                               cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string> CudaPath("cuda-path", cl::ZeroOrMore,
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

/// Binary path for the CUDA installation.
static std::string CudaBinaryPath;

/// Temporary files created by the linker wrapper.
static SmallVector<std::string, 16> TempFiles;

/// Codegen flags for LTO backend.
static codegen::RegisterCodeGenFlags CodeGenFlags;

/// Magic section string that marks the existence of offloading data. The
/// section string will be formatted as `.llvm.offloading.<triple>.<arch>`.
#define OFFLOAD_SECTION_MAGIC_STR ".llvm.offloading."

/// Information for a device offloading file extracted from the host.
struct DeviceFile {
  DeviceFile(StringRef TheTriple, StringRef Arch, StringRef Filename)
      : TheTriple(TheTriple), Arch(Arch), Filename(Filename) {}

  const std::string TheTriple;
  const std::string Arch;
  const std::string Filename;

  operator std::string() const { return TheTriple + "-" + Arch; }
};

namespace {

Expected<Optional<std::string>>
extractFromBuffer(std::unique_ptr<MemoryBuffer> Buffer,
                  SmallVectorImpl<DeviceFile> &DeviceFiles);

static StringRef getDeviceFileExtension(StringRef DeviceTriple,
                                        bool IsBitcode = false) {
  Triple TheTriple(DeviceTriple);
  if (TheTriple.isAMDGPU() || IsBitcode)
    return "bc";
  if (TheTriple.isNVPTX())
    return "cubin";
  return "o";
}

/// Extract the device file from the string '<triple>-<arch>=<library>.bc'.
DeviceFile getBitcodeLibrary(StringRef LibraryStr) {
  auto DeviceAndPath = StringRef(LibraryStr).split('=');
  auto TripleAndArch = DeviceAndPath.first.rsplit('-');
  return DeviceFile(TripleAndArch.first, TripleAndArch.second,
                    DeviceAndPath.second);
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

Error runLinker(std::string &LinkerPath, SmallVectorImpl<std::string> &Args) {
  std::vector<StringRef> LinkerArgs;
  LinkerArgs.push_back(LinkerPath);
  for (auto &Arg : Args)
    LinkerArgs.push_back(Arg);

  if (sys::ExecuteAndWait(LinkerPath, LinkerArgs))
    return createStringError(inconvertibleErrorCode(), "'linker' failed");
  return Error::success();
}

void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-linker-wrapper") << '\n';
}

void removeFromCompilerUsed(Module &M, GlobalValue &Value) {
  GlobalVariable *GV = M.getGlobalVariable("llvm.compiler.used");
  Type *Int8PtrTy = Type::getInt8PtrTy(M.getContext());
  Constant *ValueToRemove =
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(&Value, Int8PtrTy);
  SmallPtrSet<Constant *, 16> InitAsSet;
  SmallVector<Constant *, 16> Init;
  if (GV) {
    if (GV->hasInitializer()) {
      auto *CA = cast<ConstantArray>(GV->getInitializer());
      for (auto &Op : CA->operands()) {
        Constant *C = cast_or_null<Constant>(Op);
        if (C != ValueToRemove && InitAsSet.insert(C).second)
          Init.push_back(C);
      }
    }
    GV->eraseFromParent();
  }

  if (Init.empty())
    return;

  ArrayType *ATy = ArrayType::get(Int8PtrTy, Init.size());
  GV = new llvm::GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                                ConstantArray::get(ATy, Init),
                                "llvm.compiler.used");
  GV->setSection("llvm.metadata");
}

Expected<Optional<std::string>>
extractFromBinary(const ObjectFile &Obj,
                  SmallVectorImpl<DeviceFile> &DeviceFiles) {
  StringRef Extension = sys::path::extension(Obj.getFileName()).drop_front();
  StringRef Prefix = sys::path::stem(Obj.getFileName());
  SmallVector<StringRef, 4> ToBeStripped;

  // Extract data from sections of the form `.llvm.offloading.<triple>.<arch>`.
  for (const SectionRef &Sec : Obj.sections()) {
    Expected<StringRef> Name = Sec.getName();
    if (!Name || !Name->startswith(OFFLOAD_SECTION_MAGIC_STR))
      continue;

    SmallVector<StringRef, 4> SectionFields;
    Name->split(SectionFields, '.');
    StringRef DeviceTriple = SectionFields[3];
    StringRef Arch = SectionFields[4];

    if (Expected<StringRef> Contents = Sec.getContents()) {
      SmallString<128> TempFile;
      StringRef DeviceExtension = getDeviceFileExtension(
          DeviceTriple, identify_magic(*Contents) == file_magic::bitcode);
      if (Error Err =
              createOutputFile(Prefix + "-device-" + DeviceTriple + "-" + Arch,
                               DeviceExtension, TempFile))
        return std::move(Err);

      Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
          FileOutputBuffer::create(TempFile, Sec.getSize());
      if (!OutputOrErr)
        return OutputOrErr.takeError();
      std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
      std::copy(Contents->begin(), Contents->end(), Output->getBufferStart());
      if (Error E = Output->commit())
        return std::move(E);

      DeviceFiles.emplace_back(DeviceTriple, Arch, TempFile);
      ToBeStripped.push_back(*Name);
    }
  }

  if (ToBeStripped.empty() || !StripSections)
    return None;

  // If the object file to strip doesn't exist we need to write it so we can
  // pass it to llvm-strip.
  SmallString<128> StripFile = Obj.getFileName();
  if (!sys::fs::exists(StripFile)) {
    SmallString<128> TempFile;
    if (Error Err = createOutputFile(
            sys::path::stem(StripFile),
            sys::path::extension(StripFile).drop_front(), TempFile))
      return std::move(Err);

    auto Contents = Obj.getMemoryBufferRef().getBuffer();
    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(TempFile, Contents.size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    std::copy(Contents.begin(), Contents.end(), Output->getBufferStart());
    if (Error E = Output->commit())
      return std::move(E);
    StripFile = TempFile;
  }

  // We will use llvm-strip to remove the now unneeded section containing the
  // offloading code.
  ErrorOr<std::string> StripPath = sys::findProgramByName("llvm-strip");
  if (!StripPath)
    return createStringError(StripPath.getError(),
                             "Unable to find 'llvm-strip' in path");

  SmallString<128> TempFile;
  if (Error Err = createOutputFile(Prefix + "-host", Extension, TempFile))
    return std::move(Err);

  SmallVector<StringRef, 8> StripArgs;
  StripArgs.push_back(*StripPath);
  StripArgs.push_back("--no-strip-all");
  StripArgs.push_back(StripFile);
  for (auto &Section : ToBeStripped) {
    StripArgs.push_back("--remove-section");
    StripArgs.push_back(Section);
  }
  StripArgs.push_back("-o");
  StripArgs.push_back(TempFile);

  if (sys::ExecuteAndWait(*StripPath, StripArgs))
    return createStringError(inconvertibleErrorCode(), "'llvm-strip' failed");

  return static_cast<std::string>(TempFile);
}

Expected<Optional<std::string>>
extractFromBitcode(std::unique_ptr<MemoryBuffer> Buffer,
                   SmallVectorImpl<DeviceFile> &DeviceFiles) {
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = getLazyIRModule(std::move(Buffer), Err, Context);
  if (!M)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create module");

  StringRef Extension = sys::path::extension(M->getName()).drop_front();
  StringRef Prefix =
      sys::path::stem(M->getName()).take_until([](char C) { return C == '-'; });

  SmallVector<GlobalVariable *, 4> ToBeDeleted;

  // Extract data from the global string containing a section of the form
  // `.llvm.offloading.<triple>.<arch>`.
  for (GlobalVariable &GV : M->globals()) {
    if (!GV.hasSection() ||
        !GV.getSection().startswith(OFFLOAD_SECTION_MAGIC_STR))
      continue;

    auto *CDS = dyn_cast<ConstantDataSequential>(GV.getInitializer());
    if (!CDS)
      continue;

    SmallVector<StringRef, 4> SectionFields;
    GV.getSection().split(SectionFields, '.');
    StringRef DeviceTriple = SectionFields[3];
    StringRef Arch = SectionFields[4];

    StringRef Contents = CDS->getAsString();
    SmallString<128> TempFile;
    StringRef DeviceExtension = getDeviceFileExtension(
        DeviceTriple, identify_magic(Contents) == file_magic::bitcode);
    if (Error Err =
            createOutputFile(Prefix + "-device-" + DeviceTriple + "-" + Arch,
                             DeviceExtension, TempFile))
      return std::move(Err);

    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(TempFile, Contents.size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    std::copy(Contents.begin(), Contents.end(), Output->getBufferStart());
    if (Error E = Output->commit())
      return std::move(E);

    DeviceFiles.emplace_back(DeviceTriple, Arch, TempFile);
    ToBeDeleted.push_back(&GV);
  }

  if (ToBeDeleted.empty() || !StripSections)
    return None;

  // We need to materialize the lazy module before we make any changes.
  if (Error Err = M->materializeAll())
    return std::move(Err);

  // Remove the global from the module and write it to a new file.
  for (GlobalVariable *GV : ToBeDeleted) {
    removeFromCompilerUsed(*M, *GV);
    GV->eraseFromParent();
  }

  SmallString<128> TempFile;
  if (Error Err = createOutputFile(Prefix + "-host", Extension, TempFile))
    return std::move(Err);

  std::error_code EC;
  raw_fd_ostream HostOutput(TempFile, EC, sys::fs::OF_None);
  if (EC)
    return createFileError(TempFile, EC);
  WriteBitcodeToFile(*M, HostOutput);
  return static_cast<std::string>(TempFile);
}

Expected<Optional<std::string>>
extractFromArchive(const Archive &Library,
                   SmallVectorImpl<DeviceFile> &DeviceFiles) {
  bool NewMembers = false;
  SmallVector<NewArchiveMember, 8> Members;

  // Try to extract device code from each file stored in the static archive.
  // Save the stripped archive members to create a new host archive with the
  // offloading code removed.
  Error Err = Error::success();
  for (auto Child : Library.children(Err)) {
    auto ChildBufferRefOrErr = Child.getMemoryBufferRef();
    if (!ChildBufferRefOrErr)
      return ChildBufferRefOrErr.takeError();
    std::unique_ptr<MemoryBuffer> ChildBuffer =
        MemoryBuffer::getMemBuffer(*ChildBufferRefOrErr, false);

    auto FileOrErr = extractFromBuffer(std::move(ChildBuffer), DeviceFiles);
    if (!FileOrErr)
      return FileOrErr.takeError();

    // If we created a new stripped host file, use it to create a new archive
    // member, otherwise use the old member.
    if (!FileOrErr->hasValue()) {
      Expected<NewArchiveMember> NewMember =
          NewArchiveMember::getOldMember(Child, true);
      if (!NewMember)
        return NewMember.takeError();
      Members.push_back(std::move(*NewMember));
    } else {
      Expected<NewArchiveMember> NewMember =
          NewArchiveMember::getFile(**FileOrErr, true);
      if (!NewMember)
        return NewMember.takeError();
      Members.push_back(std::move(*NewMember));
      NewMembers = true;

      // We no longer need the stripped file, remove it.
      if (std::error_code EC = sys::fs::remove(**FileOrErr))
        return createFileError(**FileOrErr, EC);
    }
  }

  if (Err)
    return std::move(Err);

  if (!NewMembers || !StripSections)
    return None;

  // Create a new static library using the stripped host files.
  SmallString<128> TempFile;
  StringRef Prefix = sys::path::stem(Library.getFileName());
  if (Error Err = createOutputFile(Prefix + "-host", "a", TempFile))
    return std::move(Err);

  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(Library.getMemoryBufferRef(), false);
  if (Error Err = writeArchive(TempFile, Members, true, Library.kind(), true,
                               Library.isThin(), std::move(Buffer)))
    return std::move(Err);

  return static_cast<std::string>(TempFile);
}

/// Extracts embedded device offloading code from a memory \p Buffer to a list
/// of \p DeviceFiles. If device code was extracted a new file with the embedded
/// device code stripped from the buffer will be returned.
Expected<Optional<std::string>>
extractFromBuffer(std::unique_ptr<MemoryBuffer> Buffer,
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
    return errorCodeToError(object_error::invalid_file_type);
  }
}

// TODO: Move these to a separate file.
namespace nvptx {
Expected<std::string> assemble(StringRef InputFile, Triple TheTriple,
                               StringRef Arch) {
  // NVPTX uses the ptxas binary to create device object files.
  ErrorOr<std::string> PtxasPath =
      sys::findProgramByName("ptxas", {CudaBinaryPath});
  if (!PtxasPath)
    PtxasPath = sys::findProgramByName("ptxas");
  if (!PtxasPath)
    return createStringError(PtxasPath.getError(),
                             "Unable to find 'ptxas' in path");

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
  CmdArgs.push_back("-c");

  CmdArgs.push_back(InputFile);

  if (sys::ExecuteAndWait(*PtxasPath, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'ptxas' failed");

  return static_cast<std::string>(TempFile);
}

Expected<std::string> link(ArrayRef<std::string> InputFiles, Triple TheTriple,
                           StringRef Arch) {
  // NVPTX uses the nvlink binary to link device object files.
  ErrorOr<std::string> NvlinkPath =
      sys::findProgramByName("nvlink", {CudaBinaryPath});
  if (!NvlinkPath)
    NvlinkPath = sys::findProgramByName("nvlink");
  if (!NvlinkPath)
    return createStringError(NvlinkPath.getError(),
                             "Unable to find 'nvlink' in path");

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

  if (sys::ExecuteAndWait(*NvlinkPath, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'nvlink' failed");

  return static_cast<std::string>(TempFile);
}
} // namespace nvptx
namespace amdgcn {
Expected<std::string> link(ArrayRef<std::string> InputFiles, Triple TheTriple,
                           StringRef Arch) {
  // AMDGPU uses the lld binary to link device object files.
  ErrorOr<std::string> LLDPath =
      sys::findProgramByName("lld", sys::path::parent_path(LinkerExecutable));
  if (!LLDPath)
    LLDPath = sys::findProgramByName("lld");
  if (!LLDPath)
    return createStringError(LLDPath.getError(),
                             "Unable to find 'lld' in path");

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

  if (sys::ExecuteAndWait(*LLDPath, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'lld' failed");

  return static_cast<std::string>(TempFile);
}
} // namespace amdgcn

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
    // TODO: x86 linking support.
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
  Backend = lto::createInProcessThinBackend(
      llvm::heavyweight_hardware_concurrency(1));

  Conf.CPU = Arch.str();
  Conf.Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);

  Conf.MAttrs = getTargetFeatures(TheTriple);
  Conf.CGOptLevel = getCGOptLevel(OptLevel[1] - '0');
  Conf.OptLevel = OptLevel[1] - '0';
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
                       const Triple &TheTriple, StringRef Arch) {
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> SavedBuffers;
  SmallVector<std::unique_ptr<lto::InputFile>, 4> BitcodeFiles;
  SmallVector<std::string, 4> NewInputFiles;
  StringMap<bool> UsedInRegularObj;
  StringMap<bool> UsedInSharedLib;

  // Search for bitcode files in the input and create an LTO input file. If it
  // is not a bitcode file, scan its symbol table for symbols we need to
  // save.
  for (StringRef File : InputFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(File);
    if (std::error_code EC = BufferOrErr.getError())
      return createFileError(File, EC);

    file_magic Type = identify_magic((*BufferOrErr)->getBuffer());
    if (Type != file_magic::bitcode) {
      Expected<std::unique_ptr<ObjectFile>> ObjFile =
          ObjectFile::createObjectFile(**BufferOrErr, Type);
      if (!ObjFile)
        return ObjFile.takeError();

      NewInputFiles.push_back(File.str());
      for (auto &Sym : (*ObjFile)->symbols()) {
        Expected<StringRef> Name = Sym.getName();
        if (!Name)
          return Name.takeError();

        // Record if we've seen these symbols in any object or shared libraries.
        if ((*ObjFile)->isRelocatableObject())
          UsedInRegularObj[*Name] = true;
        else
          UsedInSharedLib[*Name] = true;
      }
    } else {
      Expected<std::unique_ptr<lto::InputFile>> InputFileOrErr =
          llvm::lto::InputFile::create(**BufferOrErr);
      if (!InputFileOrErr)
        return InputFileOrErr.takeError();

      // Save the input file and the buffer associated with its memory.
      BitcodeFiles.push_back(std::move(*InputFileOrErr));
      SavedBuffers.push_back(std::move(*BufferOrErr));
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
  bool WholeProgram = BitcodeFiles.size() == InputFiles.size();
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
          !Sym.isUndefined() && PrevailingSymbols.insert(Sym.getName()).second;

      // We need LTO to preseve the following global symbols:
      // 1) Symbols used in regular objects.
      // 2) Sections that will be given a __start/__stop symbol.
      // 3) Prevailing symbols that are needed visibile to external libraries.
      Res.VisibleToRegularObj =
          UsedInRegularObj[Sym.getName()] ||
          isValidCIdentifier(Sym.getSectionName()) ||
          (Res.Prevailing &&
           (Sym.getVisibility() != GlobalValue::HiddenVisibility &&
            !Sym.canBeOmittedFromSymbolTable()));

      // Identify symbols that must be exported dynamically and can be
      // referenced by other files.
      Res.ExportDynamic =
          Sym.getVisibility() != GlobalValue::HiddenVisibility &&
          (UsedInSharedLib[Sym.getName()] ||
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

  // Is we are compiling for NVPTX we need to run the assembler first.
  if (TheTriple.isNVPTX() && !EmbedBitcode) {
    for (auto &File : Files) {
      auto FileOrErr = nvptx::assemble(File, TheTriple, Arch);
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
                      SmallVectorImpl<std::string> &LinkedImages) {
  // Get the list of inputs for a specific device.
  StringMap<SmallVector<std::string, 4>> LinkerInputMap;
  for (auto &File : DeviceFiles)
    LinkerInputMap[StringRef(File)].push_back(File.Filename);

  // Try to link each device toolchain.
  for (auto &LinkerInput : LinkerInputMap) {
    auto TargetFeatures = LinkerInput.getKey().rsplit('-');
    Triple TheTriple(TargetFeatures.first);
    StringRef Arch(TargetFeatures.second);

    // Run LTO on any bitcode files and replace the input with the result.
    if (Error Err = linkBitcodeFiles(LinkerInput.getValue(), TheTriple, Arch))
      return Err;

    // If we are embedding bitcode for JIT, skip the final device linking.
    if (EmbedBitcode) {
      assert(!LinkerInput.getValue().empty() && "No bitcode image to embed");
      LinkedImages.push_back(LinkerInput.getValue().front());
      continue;
    }

    auto ImageOrErr = linkDevice(LinkerInput.getValue(), TheTriple, Arch);
    if (!ImageOrErr)
      return ImageOrErr.takeError();

    LinkedImages.push_back(*ImageOrErr);
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
  if (Error Err = createOutputFile(sys::path::filename(ExecutableName) +
                                       "offload-wrapper",
                                   "o", ObjectFile))
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

/// Creates the object file containing the device image and runtime registration
/// code from the device images stored in \p Images.
Expected<std::string> wrapDeviceImages(ArrayRef<std::string> Images) {
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> SavedBuffers;
  SmallVector<ArrayRef<char>, 4> ImagesToWrap;

  for (StringRef ImageFilename : Images) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ImageOrError =
        llvm::MemoryBuffer::getFileOrSTDIN(ImageFilename);
    if (std::error_code EC = ImageOrError.getError())
      return createFileError(ImageFilename, EC);
    ImagesToWrap.emplace_back((*ImageOrError)->getBufferStart(),
                              (*ImageOrError)->getBufferSize());
    SavedBuffers.emplace_back(std::move(*ImageOrError));
  }

  LLVMContext Context;
  Module M("offload.wrapper.module", Context);
  M.setTargetTriple(HostTriple);
  if (Error Err = wrapBinaries(M, ImagesToWrap))
    return std::move(Err);

  return compileModule(M);
}

Optional<std::string> findFile(StringRef Dir, const Twine &Name) {
  SmallString<128> Path;
  // TODO: Parse `--sysroot` somewhere and use it here.
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

  ExecutableName = *(llvm::find(HostLinkerArgs, "-o") + 1);
  SmallVector<std::string, 16> LinkerArgs;
  for (const std::string &Arg : HostLinkerArgs)
    LinkerArgs.push_back(Arg);

  SmallVector<StringRef, 16> LibraryPaths;
  for (StringRef Arg : LinkerArgs) {
    if (Arg.startswith("-L"))
      LibraryPaths.push_back(Arg.drop_front(2));
  }

  // Try to extract device code from the linker input and replace the linker
  // input with a new file that has the device section stripped.
  SmallVector<DeviceFile, 4> DeviceFiles;
  for (std::string &Arg : LinkerArgs) {
    if (Arg == ExecutableName)
      continue;

    // Search for static libraries in the library link path.
    std::string Filename = Arg;
    if (Optional<std::string> Library = searchLibrary(Arg, LibraryPaths))
      Filename = *Library;

    if ((sys::path::extension(Filename) == ".o" ||
         sys::path::extension(Filename) == ".a")) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
          MemoryBuffer::getFileOrSTDIN(Filename);
      if (std::error_code EC = BufferOrErr.getError())
        return reportError(createFileError(Filename, EC));

      auto NewFileOrErr =
          extractFromBuffer(std::move(*BufferOrErr), DeviceFiles);

      if (!NewFileOrErr)
        return reportError(NewFileOrErr.takeError());

      if (NewFileOrErr->hasValue())
        Arg = **NewFileOrErr;
    }
  }

  // Add the device bitcode libraries to the device files if any were passed in.
  for (StringRef LibraryStr : BitcodeLibraries)
    DeviceFiles.push_back(getBitcodeLibrary(LibraryStr));

  // Link the device images extracted from the linker input.
  SmallVector<std::string, 16> LinkedImages;
  if (Error Err = linkDeviceFiles(DeviceFiles, LinkedImages))
    return reportError(std::move(Err));

  // Wrap each linked device image into a linkable host binary and add it to the
  // link job's inputs.
  auto FileOrErr = wrapDeviceImages(LinkedImages);
  if (!FileOrErr)
    return reportError(FileOrErr.takeError());
  LinkerArgs.push_back(*FileOrErr);

  // Run the host linking job.
  if (Error Err = runLinker(LinkerUserPath, LinkerArgs))
    return reportError(std::move(Err));

  // Remove the temporary files created.
  for (const auto &TempFile : TempFiles)
    if (std::error_code EC = sys::fs::remove(TempFile))
      reportError(createFileError(TempFile, EC));

  return EXIT_SUCCESS;
}
