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

#include "clang/Basic/Version.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
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
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -help)
// will be hidden.
static cl::OptionCategory
    ClangLinkerWrapperCategory("clang-linker-wrapper options");

static cl::opt<bool> StripSections(
    "strip-sections", cl::ZeroOrMore,
    cl::desc("Strip offloading sections from the host object file."),
    cl::init(true), cl::cat(ClangLinkerWrapperCategory));

static cl::opt<std::string> LinkerUserPath("linker-path",
                                           cl::desc("Path of linker binary"),
                                           cl::cat(ClangLinkerWrapperCategory));

// Do not parse linker options.
static cl::list<std::string>
    HostLinkerArgs(cl::Sink, cl::desc("<options to be passed to linker>..."));

/// Path of the current binary.
static std::string LinkerExecutable;

/// Temporary files created by the linker wrapper.
static SmallVector<std::string, 16> TempFiles;

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
  StringRef Prefix = sys::path::stem(Obj.getFileName()).take_until([](char C) {
    return C == '-';
  });
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
      if (std::error_code EC = sys::fs::createTemporaryFile(
              Prefix + "-device-" + DeviceTriple, DeviceExtension, TempFile))
        return createFileError(TempFile, EC);
      TempFiles.push_back(static_cast<std::string>(TempFile));

      Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
          FileOutputBuffer::create(TempFile, Sec.getSize());
      if (!OutputOrErr)
        return OutputOrErr.takeError();
      std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
      std::copy(Contents->begin(), Contents->end(), Output->getBufferStart());
      if (Error E = Output->commit())
        return E;

      DeviceFiles.emplace_back(DeviceTriple, Arch, TempFile);
      ToBeStripped.push_back(*Name);
    }
  }

  if (ToBeStripped.empty())
    return None;

  // We will use llvm-strip to remove the now unneeded section containing the
  // offloading code.
  ErrorOr<std::string> StripPath = sys::findProgramByName("llvm-strip");
  if (!StripPath)
    return createStringError(StripPath.getError(),
                             "Unable to find 'llvm-strip' in path");

  SmallString<128> TempFile;
  if (std::error_code EC =
          sys::fs::createTemporaryFile(Prefix + "-host", Extension, TempFile))
    return createFileError(TempFile, EC);
  TempFiles.push_back(static_cast<std::string>(TempFile));

  SmallVector<StringRef, 8> StripArgs;
  StripArgs.push_back(*StripPath);
  StripArgs.push_back("--no-strip-all");
  StripArgs.push_back(Obj.getFileName());
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
    if (std::error_code EC = sys::fs::createTemporaryFile(
            Prefix + "-device-" + DeviceTriple, DeviceExtension, TempFile))
      return createFileError(TempFile, EC);
    TempFiles.push_back(static_cast<std::string>(TempFile));

    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(TempFile, Contents.size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    std::copy(Contents.begin(), Contents.end(), Output->getBufferStart());
    if (Error E = Output->commit())
      return E;

    DeviceFiles.emplace_back(DeviceTriple, Arch, TempFile);
    ToBeDeleted.push_back(&GV);
  }

  if (ToBeDeleted.empty())
    return None;

  // We need to materialize the lazy module before we make any changes.
  if (Error Err = M->materializeAll())
    return Err;

  // Remove the global from the module and write it to a new file.
  for (GlobalVariable *GV : ToBeDeleted) {
    removeFromCompilerUsed(*M, *GV);
    GV->eraseFromParent();
  }

  SmallString<128> TempFile;
  if (std::error_code EC =
          sys::fs::createTemporaryFile(Prefix + "-host", Extension, TempFile))
    return createFileError(TempFile, EC);
  TempFiles.push_back(static_cast<std::string>(TempFile));

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

  StringRef Extension =
      sys::path::extension(Library.getFileName()).drop_front();
  StringRef Prefix =
      sys::path::stem(Library.getFileName()).take_until([](char C) {
        return C == '-';
      });

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
    return Err;

  if (!NewMembers)
    return None;

  // Create a new static library using the stripped host files.
  SmallString<128> TempFile;
  if (std::error_code EC =
          sys::fs::createTemporaryFile(Prefix + "-host", Extension, TempFile))
    return createFileError(TempFile, EC);
  TempFiles.push_back(static_cast<std::string>(TempFile));

  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBuffer(Library.getMemoryBufferRef(), false);
  if (Error WriteErr = writeArchive(TempFile, Members, true, Library.kind(),
                                    true, Library.isThin(), std::move(Buffer)))
    return WriteErr;

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
Expected<std::string> link(ArrayRef<StringRef> InputFiles,
                           ArrayRef<std::string> LinkerArgs, Triple TheTriple,
                           StringRef Arch) {
  // NVPTX uses the nvlink binary to link device object files.
  ErrorOr<std::string> NvlinkPath = sys::findProgramByName("nvlink");
  if (!NvlinkPath)
    return createStringError(NvlinkPath.getError(),
                             "Unable to find 'nvlink' in path");

  // Create a new file to write the linked device image to.
  SmallString<128> TempFile;
  if (std::error_code EC = sys::fs::createTemporaryFile(
          TheTriple.getArchName() + "-" + Arch + "-image", "out", TempFile))
    return createFileError(TempFile, EC);
  TempFiles.push_back(static_cast<std::string>(TempFile));

  // TODO: Pass in arguments like `-g` and `-v` from the driver.
  SmallVector<StringRef, 16> CmdArgs;
  CmdArgs.push_back(*NvlinkPath);
  CmdArgs.push_back(TheTriple.isArch64Bit() ? "-m64" : "-m32");
  CmdArgs.push_back("-o");
  CmdArgs.push_back(TempFile);
  CmdArgs.push_back("-arch");
  CmdArgs.push_back(Arch);

  // Copy system library paths used by the host linker.
  for (StringRef Arg : LinkerArgs)
    if (Arg.startswith("-L"))
      CmdArgs.push_back(Arg);

  // Add extracted input files.
  for (auto Input : InputFiles)
    CmdArgs.push_back(Input);

  if (sys::ExecuteAndWait(*NvlinkPath, CmdArgs))
    return createStringError(inconvertibleErrorCode(), "'nvlink' failed");

  return static_cast<std::string>(TempFile);
}
} // namespace nvptx

Expected<std::string> linkDevice(ArrayRef<StringRef> InputFiles,
                                 ArrayRef<std::string> LinkerArgs,
                                 Triple TheTriple, StringRef Arch) {
  switch (TheTriple.getArch()) {
  case Triple::nvptx:
  case Triple::nvptx64:
    return nvptx::link(InputFiles, LinkerArgs, TheTriple, Arch);
  case Triple::amdgcn:
    // TODO: AMDGCN linking support.
  case Triple::x86:
  case Triple::x86_64:
    // TODO: x86 linking support.
  default:
    return createStringError(inconvertibleErrorCode(),
                             TheTriple.getArchName() +
                                 " linking is not supported");
  }
}

/// Runs the appropriate linking action on all the device files specified in \p
/// DeviceFiles. The linked device images are returned in \p LinkedImages.
Error linkDeviceFiles(ArrayRef<DeviceFile> DeviceFiles,
                      ArrayRef<std::string> LinkerArgs,
                      SmallVectorImpl<std::string> &LinkedImages) {
  // Get the list of inputs for a specific device.
  StringMap<SmallVector<StringRef, 4>> LinkerInputMap;
  for (auto &File : DeviceFiles)
    LinkerInputMap[StringRef(File)].push_back(File.Filename);

  // Try to link each device toolchain.
  for (auto &LinkerInput : LinkerInputMap) {
    auto TargetFeatures = LinkerInput.getKey().rsplit('-');
    Triple TheTriple(TargetFeatures.first);
    StringRef Arch(TargetFeatures.second);

    // TODO: Run LTO or bitcode linking before the final link job.

    auto ImageOrErr =
        linkDevice(LinkerInput.getValue(), LinkerArgs, TheTriple, Arch);
    if (!ImageOrErr)
      return ImageOrErr.takeError();

    LinkedImages.push_back(*ImageOrErr);
  }
  return Error::success();
}

/// Creates an object file containing the device image stored in the filename \p
/// ImageFile that can be linked with the host.
Expected<std::string> wrapDeviceImage(StringRef ImageFile) {
  // TODO: Call these utilities as a library intead of executing them here.
  ErrorOr<std::string> WrapperPath =
      sys::findProgramByName("clang-offload-wrapper");
  if (!WrapperPath)
    return createStringError(WrapperPath.getError(),
                             "Unable to find 'clang-offload-wrapper' in path");

  // Create a new file to write the wrapped bitcode file to.
  SmallString<128> BitcodeFile;
  if (std::error_code EC =
          sys::fs::createTemporaryFile("offload", "bc", BitcodeFile))
    return createFileError(BitcodeFile, EC);
  TempFiles.push_back(static_cast<std::string>(BitcodeFile));

  // TODO: Optionally pass the host triple in somewhere.
  Triple HostTriple(sys::getDefaultTargetTriple());
  SmallVector<StringRef, 4> WrapperArgs;
  WrapperArgs.push_back(*WrapperPath);
  WrapperArgs.push_back("-target");
  WrapperArgs.push_back(HostTriple.getTriple());
  WrapperArgs.push_back("-o");
  WrapperArgs.push_back(BitcodeFile);
  WrapperArgs.push_back(ImageFile);

  if (sys::ExecuteAndWait(*WrapperPath, WrapperArgs))
    return createStringError(inconvertibleErrorCode(),
                             "'clang-offload-wrapper' failed");

  ErrorOr<std::string> CompilerPath = sys::findProgramByName("llc");
  if (!WrapperPath)
    return createStringError(WrapperPath.getError(),
                             "Unable to find 'llc' in path");

  // Create a new file to write the wrapped bitcode file to.
  SmallString<128> ObjectFile;
  if (std::error_code EC =
          sys::fs::createTemporaryFile("offload", "o", ObjectFile))
    return createFileError(BitcodeFile, EC);
  TempFiles.push_back(static_cast<std::string>(ObjectFile));

  SmallVector<StringRef, 4> CompilerArgs;
  CompilerArgs.push_back(*CompilerPath);
  CompilerArgs.push_back("--filetype=obj");
  CompilerArgs.push_back("--relocation-model=pic");
  CompilerArgs.push_back("-o");
  CompilerArgs.push_back(ObjectFile);
  CompilerArgs.push_back(BitcodeFile);

  if (sys::ExecuteAndWait(*CompilerPath, CompilerArgs))
    return createStringError(inconvertibleErrorCode(), "'llc' failed");

  return static_cast<std::string>(ObjectFile);
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

  SmallVector<std::string, 16> LinkerArgs;
  for (const std::string &Arg : HostLinkerArgs)
    LinkerArgs.push_back(Arg);

  SmallVector<StringRef, 16> LibraryPaths;
  for (const StringRef Arg : LinkerArgs)
    if (Arg.startswith("-L"))
      LibraryPaths.push_back(Arg.drop_front(2));

  // Try to extract device code from the linker input and replace the linker
  // input with a new file that has the device section stripped.
  SmallVector<DeviceFile, 4> DeviceFiles;
  for (std::string &Arg : LinkerArgs) {
    // Search for static libraries in the library link path.
    std::string Filename = Arg;
    if (Optional<std::string> Library = searchLibrary(Arg, LibraryPaths))
      Filename = *Library;

    if (sys::path::extension(Filename) == ".o" ||
        sys::path::extension(Filename) == ".a") {
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

  // Link the device images extracted from the linker input.
  SmallVector<std::string, 16> LinkedImages;
  if (Error Err = linkDeviceFiles(DeviceFiles, LinkerArgs, LinkedImages))
    return reportError(std::move(Err));

  // Wrap each linked device image into a linkable host binary and add it to the
  // link job's inputs.
  for (const auto &Image : LinkedImages) {
    auto FileOrErr = wrapDeviceImage(Image);
    if (!FileOrErr)
      return reportError(FileOrErr.takeError());

    LinkerArgs.push_back(*FileOrErr);
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
