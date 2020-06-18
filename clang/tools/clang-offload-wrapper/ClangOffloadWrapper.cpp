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
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

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

namespace {

class BinaryWrapper {
  LLVMContext C;
  Module M;

  StructType *EntryTy = nullptr;
  StructType *ImageTy = nullptr;
  StructType *DescTy = nullptr;

private:
  IntegerType *getSizeTTy() {
    switch (M.getDataLayout().getPointerTypeSize(Type::getInt8PtrTy(C))) {
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
      EntryTy = StructType::create("__tgt_offload_entry", Type::getInt8PtrTy(C),
                                   Type::getInt8PtrTy(C), getSizeTTy(),
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
      ImageTy = StructType::create("__tgt_device_image", Type::getInt8PtrTy(C),
                                   Type::getInt8PtrTy(C), getEntryPtrTy(),
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
        M, DummyInit->getType(), true, GlobalVariable::ExternalLinkage,
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

  void createRegisterFunction(GlobalVariable *BinDesc) {
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
    // __tgt_register_requires (we want to know what requirements have been
    // asked for before we load a libomptarget plugin so that by the time the
    // plugin is loaded it can report how many devices there are which can
    // satisfy these requirements).
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
  BinaryWrapper(StringRef Target) : M("offload.wrapper.object", C) {
    M.setTargetTriple(Target);
  }

  const Module &wrapBinaries(ArrayRef<ArrayRef<char>> Binaries) {
    GlobalVariable *Desc = createBinDesc(Binaries);
    assert(Desc && "no binary descriptor");
    createRegisterFunction(Desc);
    createUnregisterFunction(Desc);
    return M;
  }
};

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

  // Read device binaries.
  SmallVector<std::unique_ptr<MemoryBuffer>, 4u> Buffers;
  SmallVector<ArrayRef<char>, 4u> Images;
  Buffers.reserve(Inputs.size());
  Images.reserve(Inputs.size());
  for (const std::string &File : Inputs) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFileOrSTDIN(File);
    if (!BufOrErr) {
      reportError(createFileError(File, BufOrErr.getError()));
      return 1;
    }
    const std::unique_ptr<MemoryBuffer> &Buf =
        Buffers.emplace_back(std::move(*BufOrErr));
    Images.emplace_back(Buf->getBufferStart(), Buf->getBufferSize());
  }

  // Create the output file to write the resulting bitcode to.
  std::error_code EC;
  ToolOutputFile Out(Output, EC, sys::fs::OF_None);
  if (EC) {
    reportError(createFileError(Output, EC));
    return 1;
  }

  // Create a wrapper for device binaries and write its bitcode to the file.
  WriteBitcodeToFile(BinaryWrapper(Target).wrapBinaries(
                         makeArrayRef(Images.data(), Images.size())),
                     Out.os());
  if (Out.os().has_error()) {
    reportError(createFileError(Output, Out.os().error()));
    return 1;
  }

  // Success.
  Out.keep();
  return 0;
}
