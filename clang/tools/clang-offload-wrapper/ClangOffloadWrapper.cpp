//===-- clang-offload-wrapper/ClangOffloadWrapper.cpp -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the offload wrapper tool. It takes offload target binaries
/// as input and creates wrapper bitcode file containing target binaries
/// packaged as data. Wrapper bitcode also includes initialization code which
/// registers target binaries in offloading runtime at program startup.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VCSRevision.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cassert>
#include <cstdint>

#define OPENMP_OFFLOAD_IMAGE_VERSION "1.0"

using namespace llvm;
using namespace llvm::object;
using OffloadingImage = OffloadBinary::OffloadingImage;

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

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadWrapperCategory("clang-offload-wrapper options");

static cl::opt<std::string> Output("o", cl::Required,
                                   cl::desc("Output filename"),
                                   cl::value_desc("filename"),
                                   cl::cat(ClangOffloadWrapperCategory));

static cl::list<std::string> Inputs(cl::Positional, cl::OneOrMore,
                                    cl::desc("<input files>"),
                                    cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string>
    Target("target", cl::Required,
           cl::desc("Target triple for the output module"),
           cl::value_desc("triple"), cl::cat(ClangOffloadWrapperCategory));

static cl::opt<bool> SaveTemps(
    "save-temps",
    cl::desc("Save temporary files that may be produced by the tool. "
             "This option forces print-out of the temporary files' names."),
    cl::Hidden);

static cl::opt<bool> AddOpenMPOffloadNotes(
    "add-omp-offload-notes",
    cl::desc("Add LLVMOMPOFFLOAD ELF notes to ELF device images."), cl::Hidden);

static cl::list<std::string>
    OffloadArch("offload-arch",
                cl::desc("Contains offload-arch of the following target binary."),
                cl::value_desc("offload-arch-name"),
                cl::cat(ClangOffloadWrapperCategory));

namespace {

class BinaryWrapper {
  LLVMContext C;
  Module M;

  StructType *EntryTy = nullptr;
  StructType *ImageTy = nullptr;
  StructType *DescTy = nullptr;

  std::string ToolName;
  std::string ObjcopyPath;
  // Temporary file names that may be created during adding notes
  // to ELF offload images. Use -save-temps to keep them and also
  // see their names. A temporary file's name includes the name
  // of the original input ELF image, so you can easily match
  // them, if you have multiple inputs.
  std::vector<std::string> TempFiles;

private:
  IntegerType *getSizeTTy() {
    switch (M.getDataLayout().getPointerTypeSize(PointerType::getUnqual(C))) {
    case 4u:
      return Type::getInt32Ty(C);
    case 8u:
      return Type::getInt64Ty(C);
    }
    llvm_unreachable("unsupported pointer type size");
  }

  // struct __tgt_offload_entry {
  //   void *addr;
  //   char *name;
  //   size_t size;
  //   int32_t flags;
  //   int32_t reserved;
  // };
  StructType *getEntryTy() {
    if (!EntryTy)
      EntryTy = StructType::create("__tgt_offload_entry", PointerType::getUnqual(C),
                                   PointerType::getUnqual(C), getSizeTTy(),
                                   Type::getInt32Ty(C), Type::getInt32Ty(C));
    return EntryTy;
  }

  PointerType *getEntryPtrTy() { return PointerType::getUnqual(getEntryTy()); }

  // struct __tgt_device_image {
  //   void *ImageStart;
  //   void *ImageEnd;
  //   __tgt_offload_entry *EntriesBegin;
  //   __tgt_offload_entry *EntriesEnd;
  // };
  StructType *getDeviceImageTy() {
    if (!ImageTy)
      ImageTy = StructType::create("__tgt_device_image", PointerType::getUnqual(C),
                                   PointerType::getUnqual(C), getEntryPtrTy(),
                                   getEntryPtrTy());
    return ImageTy;
  }

  PointerType *getDeviceImagePtrTy() {
    return PointerType::getUnqual(getDeviceImageTy());
  }

  // struct __tgt_bin_desc {
  //   int32_t NumDeviceImages;
  //   __tgt_device_image *DeviceImages;
  //   __tgt_offload_entry *HostEntriesBegin;
  //   __tgt_offload_entry *HostEntriesEnd;
  // };
  StructType *getBinDescTy() {
    if (!DescTy)
      DescTy = StructType::create("__tgt_bin_desc", Type::getInt32Ty(C),
                                  getDeviceImagePtrTy(), getEntryPtrTy(),
                                  getEntryPtrTy());
    return DescTy;
  }

  PointerType *getBinDescPtrTy() {
    return PointerType::getUnqual(getBinDescTy());
  }

  /// Creates binary descriptor for the given device images. Binary descriptor
  /// is an object that is passed to the offloading runtime at program startup
  /// and it describes all device images available in the executable or shared
  /// library. It is defined as follows
  ///
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__start_omp_offloading_entries;
  /// __attribute__((visibility("hidden")))
  /// extern __tgt_offload_entry *__stop_omp_offloading_entries;
  ///
  /// static const char Image0[] = { <Bufs.front() contents> };
  ///  ...
  /// static const char ImageN[] = { <Bufs.back() contents> };
  ///
  /// static const __tgt_device_image Images[] = {
  ///   {
  ///     Image0,                            /*ImageStart*/
  ///     Image0 + sizeof(Image0),           /*ImageEnd*/
  ///     __start_omp_offloading_entries,    /*EntriesBegin*/
  ///     __stop_omp_offloading_entries      /*EntriesEnd*/
  ///   },
  ///   ...
  ///   {
  ///     ImageN,                            /*ImageStart*/
  ///     ImageN + sizeof(ImageN),           /*ImageEnd*/
  ///     __start_omp_offloading_entries,    /*EntriesBegin*/
  ///     __stop_omp_offloading_entries      /*EntriesEnd*/
  ///   }
  /// };
  ///
  /// static const __tgt_bin_desc BinDesc = {
  ///   sizeof(Images) / sizeof(Images[0]),  /*NumDeviceImages*/
  ///   Images,                              /*DeviceImages*/
  ///   __start_omp_offloading_entries,      /*HostEntriesBegin*/
  ///   __stop_omp_offloading_entries        /*HostEntriesEnd*/
  /// };
  ///
  /// Global variable that represents BinDesc is returned.
  GlobalVariable *createBinDesc(ArrayRef<ArrayRef<char>> Bufs) {
    // Create external begin/end symbols for the offload entries table.
    auto *EntriesB = new GlobalVariable(
        M, getEntryTy(), /*isConstant*/ true, GlobalValue::ExternalLinkage,
        /*Initializer*/ nullptr, "__start_omp_offloading_entries");
    EntriesB->setVisibility(GlobalValue::HiddenVisibility);
    auto *EntriesE = new GlobalVariable(
        M, getEntryTy(), /*isConstant*/ true, GlobalValue::ExternalLinkage,
        /*Initializer*/ nullptr, "__stop_omp_offloading_entries");
    EntriesE->setVisibility(GlobalValue::HiddenVisibility);

    // We assume that external begin/end symbols that we have created above will
    // be defined by the linker. But linker will do that only if linker inputs
    // have section with "omp_offloading_entries" name which is not guaranteed.
    // So, we just create dummy zero sized object in the offload entries section
    // to force linker to define those symbols.
    auto *DummyInit =
        ConstantAggregateZero::get(ArrayType::get(getEntryTy(), 0u));
    auto *DummyEntry = new GlobalVariable(
        M, DummyInit->getType(), true,
        Triple(Target).isAMDGCN() ?
          GlobalVariable::WeakAnyLinkage :
          GlobalVariable::ExternalLinkage,
        DummyInit, "__dummy.omp_offloading.entry");
    DummyEntry->setSection("omp_offloading_entries");
    DummyEntry->setVisibility(GlobalValue::HiddenVisibility);

    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    Constant *ZeroZero[] = {Zero, Zero};

    // Create initializer for the images array.
    SmallVector<Constant *, 4u> ImagesInits;
    ImagesInits.reserve(Bufs.size());
    for (ArrayRef<char> Buf : Bufs) {
      auto *Data = ConstantDataArray::get(C, Buf);
      auto *Image = new GlobalVariable(M, Data->getType(), /*isConstant*/ true,
                                       GlobalVariable::InternalLinkage, Data,
                                       ".omp_offloading.device_image");
      Image->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
      Image->setSection(".llvm.offloading");
      Image->setAlignment(Align(object::OffloadBinary::getAlignment()));

      auto *Size = ConstantInt::get(getSizeTTy(), Buf.size());
      Constant *ZeroSize[] = {Zero, Size};

      auto *ImageB = ConstantExpr::getGetElementPtr(Image->getValueType(),
                                                    Image, ZeroZero);
      auto *ImageE = ConstantExpr::getGetElementPtr(Image->getValueType(),
                                                    Image, ZeroSize);

      ImagesInits.push_back(ConstantStruct::get(getDeviceImageTy(), ImageB,
                                                ImageE, EntriesB, EntriesE));
    }

    // Then create images array.
    auto *ImagesData = ConstantArray::get(
        ArrayType::get(getDeviceImageTy(), ImagesInits.size()), ImagesInits);

    auto *Images =
        new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                           GlobalValue::InternalLinkage, ImagesData,
                           ".omp_offloading.device_images");
    Images->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    auto *ImagesB = ConstantExpr::getGetElementPtr(Images->getValueType(),
                                                   Images, ZeroZero);

    // And finally create the binary descriptor object.
    auto *DescInit = ConstantStruct::get(
        getBinDescTy(),
        ConstantInt::get(Type::getInt32Ty(C), ImagesInits.size()), ImagesB,
        EntriesB, EntriesE);

    return new GlobalVariable(M, DescInit->getType(), /*isConstant*/ true,
                              GlobalValue::InternalLinkage, DescInit,
                              ".omp_offloading.descriptor");
  }

  void createRegisterFunction(GlobalVariable *BinDesc,
                              ArrayRef<ArrayRef<char>> OffloadArchs) {

    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  ".omp_offloading.descriptor_reg", &M);
    Func->setSection(".text.startup");

    // Get __tgt_register_lib function declaration.
    auto *RegFuncTy = FunctionType::get(Type::getVoidTy(C), getBinDescPtrTy(),
                                        /*isVarArg*/ false);
    FunctionCallee RegFuncC =
        M.getOrInsertFunction("__tgt_register_lib", RegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(RegFuncC, BinDesc);
    Builder.CreateRetVoid();

    // Add this function to constructors.
    // Set priority to 1 so that __tgt_register_lib is executed AFTER
    // __tgt_register_requires (we want to know what offload-arch have been
    // asked for before we load a libomptarget plugin so that by the time the
    // plugin is loaded it can report how many devices there are which can
    // match with the offload-arch).
    appendToGlobalCtors(M, Func, /*Priority*/ 1);
  }

  void createUnregisterFunction(GlobalVariable *BinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  ".omp_offloading.descriptor_unreg", &M);
    Func->setSection(".text.startup");

    // Get __tgt_unregister_lib function declaration.
    auto *UnRegFuncTy = FunctionType::get(Type::getVoidTy(C), getBinDescPtrTy(),
                                          /*isVarArg*/ false);
    FunctionCallee UnRegFuncC =
        M.getOrInsertFunction("__tgt_unregister_lib", UnRegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(UnRegFuncC, BinDesc);
    Builder.CreateRetVoid();

    // Add this function to global destructors.
    // Match priority of __tgt_register_lib
    appendToGlobalDtors(M, Func, /*Priority*/ 1);
  }

public:
  BinaryWrapper(StringRef Target, StringRef ToolName)
      : M("offload.wrapper.object", C), ToolName(ToolName) {
    M.setTargetTriple(Target);
    // Look for llvm-objcopy in the same directory, from which
    // clang-offload-wrapper is invoked. This helps OpenMP offload
    // LIT tests.

    // This just needs to be some symbol in the binary; C++ doesn't
    // allow taking the address of ::main however.
    void *P = (void *)(intptr_t)&Help;
    std::string COWPath = sys::fs::getMainExecutable(ToolName.str().c_str(), P);
    if (!COWPath.empty()) {
      auto COWDir = sys::path::parent_path(COWPath);
      ErrorOr<std::string> ObjcopyPathOrErr =
          sys::findProgramByName("llvm-objcopy", {COWDir});
      if (ObjcopyPathOrErr) {
        ObjcopyPath = *ObjcopyPathOrErr;
        return;
      }

      // Otherwise, look through PATH environment.
    }

    ErrorOr<std::string> ObjcopyPathOrErr =
        sys::findProgramByName("llvm-objcopy");
    if (!ObjcopyPathOrErr) {
      WithColor::warning(errs(), ToolName)
          << "cannot find llvm-objcopy[.exe] in PATH; ELF notes cannot be "
             "added.\n";
      return;
    }

    ObjcopyPath = *ObjcopyPathOrErr;
  }

  ~BinaryWrapper() {
    if (TempFiles.empty())
      return;

    StringRef ToolNameRef(ToolName);
    auto warningOS = [ToolNameRef]() -> raw_ostream & {
      return WithColor::warning(errs(), ToolNameRef);
    };

    for (auto &F : TempFiles) {
      if (SaveTemps) {
        warningOS() << "keeping temporary file " << F << "\n";
        continue;
      }

      auto EC = sys::fs::remove(F, false);
      if (EC)
        warningOS() << "cannot remove temporary file " << F << ": "
                    << EC.message().c_str() << "\n";
    }
  }

  const Module &wrapBinaries(ArrayRef<ArrayRef<char>> Binaries,
                             ArrayRef<ArrayRef<char>> OffloadArchs) {
    GlobalVariable *Desc = createBinDesc(Binaries);
    assert(Desc && "no binary descriptor");
    createRegisterFunction(Desc, OffloadArchs);
    createUnregisterFunction(Desc);
    return M;
  }

  std::unique_ptr<MemoryBuffer> addELFNotes(std::unique_ptr<MemoryBuffer> Buf,
                                            StringRef OriginalFileName) {
    // Cannot add notes, if llvm-objcopy is not available.
    //
    // I did not find a clean way to add a new notes section into an existing
    // ELF file. llvm-objcopy seems to recreate a new ELF from scratch,
    // and we just try to use llvm-objcopy here.
    if (ObjcopyPath.empty())
      return Buf;

    StringRef ToolNameRef(ToolName);

    // Helpers to emit warnings.
    auto warningOS = [ToolNameRef]() -> raw_ostream & {
      return WithColor::warning(errs(), ToolNameRef);
    };
    auto handleErrorAsWarning = [&warningOS](Error E) {
      logAllUnhandledErrors(std::move(E), warningOS());
    };

    Expected<std::unique_ptr<ObjectFile>> BinOrErr =
        ObjectFile::createELFObjectFile(Buf->getMemBufferRef(),
                                        /*InitContent=*/false);
    if (Error E = BinOrErr.takeError()) {
      consumeError(std::move(E));
      // This warning is questionable, but let it be here,
      // assuming that most OpenMP offload models use ELF offload images.
      warningOS() << OriginalFileName
                  << " is not an ELF image, so notes cannot be added to it.\n";
      return Buf;
    }

    // If we fail to add the note section, we just pass through the original
    // ELF image for wrapping. At some point we should enforce the note section
    // and start emitting errors vs warnings.
    llvm::endianness Endianness;
    if (isa<ELF64LEObjectFile>(BinOrErr->get()) ||
        isa<ELF32LEObjectFile>(BinOrErr->get())) {
      Endianness = llvm::endianness::little;
    } else if (isa<ELF64BEObjectFile>(BinOrErr->get()) ||
               isa<ELF32BEObjectFile>(BinOrErr->get())) {
      Endianness = llvm::endianness::big;
    } else {
      warningOS() << OriginalFileName
                  << " is an ELF image of unrecognized format.\n";
      return Buf;
    }

    // Create temporary file for the data of a new SHT_NOTE section.
    // We fill it in with data and then pass to llvm-objcopy invocation
    // for reading.
    Twine NotesFileModel = OriginalFileName + Twine(".elfnotes.%%%%%%%.tmp");
    Expected<sys::fs::TempFile> NotesTemp =
        sys::fs::TempFile::create(NotesFileModel);
    if (Error E = NotesTemp.takeError()) {
      handleErrorAsWarning(createFileError(NotesFileModel, std::move(E)));
      return Buf;
    }
    TempFiles.push_back(NotesTemp->TmpName);

    // Create temporary file for the updated ELF image.
    // This is an empty file that we pass to llvm-objcopy invocation
    // for writing.
    Twine ELFFileModel = OriginalFileName + Twine(".elfwithnotes.%%%%%%%.tmp");
    Expected<sys::fs::TempFile> ELFTemp =
        sys::fs::TempFile::create(ELFFileModel);
    if (Error E = ELFTemp.takeError()) {
      handleErrorAsWarning(createFileError(ELFFileModel, std::move(E)));
      return Buf;
    }
    TempFiles.push_back(ELFTemp->TmpName);

    // Keep the new ELF image file to reserve the name for the future
    // llvm-objcopy invocation.
    std::string ELFTmpFileName = ELFTemp->TmpName;
    if (Error E = ELFTemp->keep(ELFTmpFileName)) {
      handleErrorAsWarning(createFileError(ELFTmpFileName, std::move(E)));
      return Buf;
    }

    // Write notes to the *elfnotes*.tmp file.
    raw_fd_ostream NotesOS(NotesTemp->FD, false);

    struct NoteTy {
      // Note name is a null-terminated "LLVMOMPOFFLOAD".
      std::string Name;
      // Note type defined in llvm/include/llvm/BinaryFormat/ELF.h.
      uint32_t Type = 0;
      // Each note has type-specific associated data.
      std::string Desc;

      NoteTy(std::string &&Name, uint32_t Type, std::string &&Desc)
          : Name(std::move(Name)), Type(Type), Desc(std::move(Desc)) {}
    };

    // So far we emit just three notes.
    SmallVector<NoteTy, 3> Notes;
    // Version of the offload image identifying the structure of the ELF image.
    // Version 1.0 does not have any specific requirements.
    // We may come up with some structure that has to be honored by all
    // offload implementations in future (e.g. to let libomptarget
    // get some information from the offload image).
    Notes.emplace_back("LLVMOMPOFFLOAD", ELF::NT_LLVM_OPENMP_OFFLOAD_VERSION,
                       OPENMP_OFFLOAD_IMAGE_VERSION);
    // This is a producer identification string. We are LLVM!
    Notes.emplace_back("LLVMOMPOFFLOAD", ELF::NT_LLVM_OPENMP_OFFLOAD_PRODUCER,
                       "LLVM");
    // This is a producer version. Use the same format that is used
    // by clang to report the LLVM version.
    Notes.emplace_back("LLVMOMPOFFLOAD",
                       ELF::NT_LLVM_OPENMP_OFFLOAD_PRODUCER_VERSION,
                       LLVM_VERSION_STRING
#ifdef LLVM_REVISION
                       " " LLVM_REVISION
#endif
    );

    // Return the amount of padding required for a blob of N bytes
    // to be aligned to Alignment bytes.
    auto getPadAmount = [](uint32_t N, uint32_t Alignment) -> uint32_t {
      uint32_t Mod = (N % Alignment);
      if (Mod == 0)
        return 0;
      return Alignment - Mod;
    };
    auto emitPadding = [&getPadAmount](raw_ostream &OS, uint32_t Size) {
      for (uint32_t I = 0; I < getPadAmount(Size, 4); ++I)
        OS << '\0';
    };

    // Put notes into the file.
    for (auto &N : Notes) {
      assert(!N.Name.empty() && "We should not create notes with empty names.");
      // Name must be null-terminated.
      if (N.Name.back() != '\0')
        N.Name += '\0';
      uint32_t NameSz = N.Name.size();
      uint32_t DescSz = N.Desc.size();
      // A note starts with three 4-byte values:
      //   NameSz
      //   DescSz
      //   Type
      // These three fields are endian-sensitive.
      support::endian::write<uint32_t>(NotesOS, NameSz, Endianness);
      support::endian::write<uint32_t>(NotesOS, DescSz, Endianness);
      support::endian::write<uint32_t>(NotesOS, N.Type, Endianness);
      // Next, we have a null-terminated Name padded to a 4-byte boundary.
      NotesOS << N.Name;
      emitPadding(NotesOS, NameSz);
      if (DescSz == 0)
        continue;
      // Finally, we have a descriptor, which is an arbitrary flow of bytes.
      NotesOS << N.Desc;
      emitPadding(NotesOS, DescSz);
    }
    NotesOS.flush();

    // Keep the notes file.
    std::string NotesTmpFileName = NotesTemp->TmpName;
    if (Error E = NotesTemp->keep(NotesTmpFileName)) {
      handleErrorAsWarning(createFileError(NotesTmpFileName, std::move(E)));
      return Buf;
    }

    // Run llvm-objcopy like this:
    //   llvm-objcopy --add-section=.note.openmp=<notes-tmp-file-name> \
    //       <orig-file-name> <elf-tmp-file-name>
    //
    // This will add a SHT_NOTE section on top of the original ELF.
    std::vector<StringRef> Args;
    Args.push_back(ObjcopyPath);
    std::string Option("--add-section=.note.openmp=" + NotesTmpFileName);
    Args.push_back(Option);
    Args.push_back(OriginalFileName);
    Args.push_back(ELFTmpFileName);
    bool ExecutionFailed = false;
    std::string ErrMsg;
    (void)sys::ExecuteAndWait(ObjcopyPath, Args,
                              /*Env=*/std::nullopt, /*Redirects=*/{},
                              /*SecondsToWait=*/0,
                              /*MemoryLimit=*/0, &ErrMsg, &ExecutionFailed);

    if (ExecutionFailed) {
      warningOS() << ErrMsg << "\n";
      return Buf;
    }

    // Substitute the original ELF with new one.
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(ELFTmpFileName);
    if (!BufOrErr) {
      handleErrorAsWarning(
          createFileError(ELFTmpFileName, BufOrErr.getError()));
      return Buf;
    }

    return std::move(*BufOrErr);
  }
};

Expected<SmallVector<std::unique_ptr<MemoryBuffer>>>
bundleImage(ArrayRef<OffloadingImage> Images) {
  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  for (const OffloadingImage &Image : Images)
    Buffers.emplace_back(
	MemoryBuffer::getMemBufferCopy(OffloadBinary::write(Image)));

  return std::move(Buffers);
}

} // anonymous namespace

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadWrapperCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-wrapper") << '\n';
  });
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to create a wrapper bitcode for offload target binaries. Takes "
      "offload\ntarget binaries as input and produces bitcode file containing "
      "target binaries packaged\nas data and initialization code which "
      "registers target binaries in offload runtime.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
  };

  if (Triple(Target).getArch() == Triple::UnknownArch) {
    reportError(createStringError(
        errc::invalid_argument, "'" + Target + "': unsupported target triple"));
    return 1;
  }

  BinaryWrapper Wrapper(Target, argv[0]);

  // Collect offload-archs.
  SmallVector<ArrayRef<char>, 4u> OffloadArchs;
  OffloadArchs.reserve(OffloadArch.size());
  for (unsigned i = 0; i != OffloadArch.size(); ++i) {
    OffloadArch[i].append("\0");
    OffloadArchs.emplace_back(OffloadArch[i].data(), OffloadArch[i].size() + 1);
  }

  // Create the output file to write the resulting bitcode to.
  std::error_code EC;
  ToolOutputFile Out(Output, EC, sys::fs::OF_None);
  if (EC) {
    reportError(createFileError(Output, EC));
    return 1;
  }

  // Read device binaries.
  DenseMap<OffloadKind, SmallVector<OffloadingImage>> Images;
  SmallVector<ArrayRef<char>, 4> BuffersToWrap;

  int numOffloadArch = 0;
  for (const std::string &File : Inputs) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFileOrSTDIN(File);
    if (!BufOrErr) {
      reportError(createFileError(File, BufOrErr.getError()));
      return 1;
    }
    std::unique_ptr<MemoryBuffer> Buffer(std::move(*BufOrErr));
    if (File != "-" && AddOpenMPOffloadNotes) {
      // Adding ELF notes for STDIN is not supported yet.
      Buffer = Wrapper.addELFNotes(std::move(Buffer), File);
    }

   OffloadingImage TheImage{};
   TheImage.TheImageKind = IMG_Bitcode;
   TheImage.TheOffloadKind = OFK_OpenMP ;
   TheImage.StringData["triple"] =
     Triple(Target).isAMDGCN() ? "amdgcn-amd-amdhsa" : "nvptx64-nvidia-cuda";
   if(OffloadArchs.size() != 0){
     TheImage.StringData["arch"] =
       OffloadArch[numOffloadArch].c_str();
     numOffloadArch++;
   } else
     TheImage.StringData["arch"] = "";
   TheImage.Image = std::move(Buffer);
   Images[OFK_OpenMP].emplace_back(std::move(TheImage));
  }

  // Bundle and wrap binaries
  for (auto &[Kind, Input] : Images) {
    // We sort the entries before bundling so they appear in a deterministic
    // order in the final binary.
    llvm::sort(Input, [](OffloadingImage &A, OffloadingImage &B) {
      return A.StringData["triple"] > B.StringData["triple"] ||
             A.StringData["arch"] > B.StringData["arch"] ||
             A.TheOffloadKind < B.TheOffloadKind;
    });
    auto BundledImagesOrErr = bundleImage(Input);
    if (!BundledImagesOrErr)
      return 1;
    for (const auto &myImage : *BundledImagesOrErr)
    BuffersToWrap.emplace_back(
      ArrayRef<char>(myImage->getBufferStart(), myImage->getBufferSize()));
    // Create a wrapper for device binaries and write its bitcode to the file.
    WriteBitcodeToFile(
      Wrapper.wrapBinaries(BuffersToWrap,
                           ArrayRef(OffloadArchs.data(), OffloadArchs.size())),
                           Out.os());
    if (Out.os().has_error()) {
      reportError(createFileError(Output, Out.os().error()));
      return 1;
    }
  }

  // Success.
  Out.keep();
  return 0;
}
