//===- OffloadWrapper.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OffloadWrapper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {
/// Magic number that begins the section containing the CUDA fatbinary.
constexpr unsigned CudaFatMagic = 0x466243b1;

IntegerType *getSizeTTy(Module &M) {
  LLVMContext &C = M.getContext();
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
StructType *getEntryTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *EntryTy = StructType::getTypeByName(C, "__tgt_offload_entry");
  if (!EntryTy)
    EntryTy = StructType::create("__tgt_offload_entry", Type::getInt8PtrTy(C),
                                 Type::getInt8PtrTy(C), getSizeTTy(M),
                                 Type::getInt32Ty(C), Type::getInt32Ty(C));
  return EntryTy;
}

PointerType *getEntryPtrTy(Module &M) {
  return PointerType::getUnqual(getEntryTy(M));
}

// struct __tgt_device_image {
//   void *ImageStart;
//   void *ImageEnd;
//   __tgt_offload_entry *EntriesBegin;
//   __tgt_offload_entry *EntriesEnd;
// };
StructType *getDeviceImageTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *ImageTy = StructType::getTypeByName(C, "__tgt_device_image");
  if (!ImageTy)
    ImageTy = StructType::create("__tgt_device_image", Type::getInt8PtrTy(C),
                                 Type::getInt8PtrTy(C), getEntryPtrTy(M),
                                 getEntryPtrTy(M));
  return ImageTy;
}

PointerType *getDeviceImagePtrTy(Module &M) {
  return PointerType::getUnqual(getDeviceImageTy(M));
}

// struct __tgt_bin_desc {
//   int32_t NumDeviceImages;
//   __tgt_device_image *DeviceImages;
//   __tgt_offload_entry *HostEntriesBegin;
//   __tgt_offload_entry *HostEntriesEnd;
// };
StructType *getBinDescTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *DescTy = StructType::getTypeByName(C, "__tgt_bin_desc");
  if (!DescTy)
    DescTy = StructType::create("__tgt_bin_desc", Type::getInt32Ty(C),
                                getDeviceImagePtrTy(M), getEntryPtrTy(M),
                                getEntryPtrTy(M));
  return DescTy;
}

PointerType *getBinDescPtrTy(Module &M) {
  return PointerType::getUnqual(getBinDescTy(M));
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
GlobalVariable *createBinDesc(Module &M, ArrayRef<ArrayRef<char>> Bufs) {
  LLVMContext &C = M.getContext();
  // Create external begin/end symbols for the offload entries table.
  auto *EntriesB = new GlobalVariable(
      M, getEntryTy(M), /*isConstant*/ true, GlobalValue::ExternalLinkage,
      /*Initializer*/ nullptr, "__start_omp_offloading_entries");
  EntriesB->setVisibility(GlobalValue::HiddenVisibility);
  auto *EntriesE = new GlobalVariable(
      M, getEntryTy(M), /*isConstant*/ true, GlobalValue::ExternalLinkage,
      /*Initializer*/ nullptr, "__stop_omp_offloading_entries");
  EntriesE->setVisibility(GlobalValue::HiddenVisibility);

  // We assume that external begin/end symbols that we have created above will
  // be defined by the linker. But linker will do that only if linker inputs
  // have section with "omp_offloading_entries" name which is not guaranteed.
  // So, we just create dummy zero sized object in the offload entries section
  // to force linker to define those symbols.
  auto *DummyInit =
      ConstantAggregateZero::get(ArrayType::get(getEntryTy(M), 0u));
  auto *DummyEntry = new GlobalVariable(
      M, DummyInit->getType(), true, GlobalVariable::ExternalLinkage, DummyInit,
      "__dummy.omp_offloading.entry");
  DummyEntry->setSection("omp_offloading_entries");
  DummyEntry->setVisibility(GlobalValue::HiddenVisibility);

  auto *Zero = ConstantInt::get(getSizeTTy(M), 0u);
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

    auto *Size = ConstantInt::get(getSizeTTy(M), Buf.size());
    Constant *ZeroSize[] = {Zero, Size};

    auto *ImageB =
        ConstantExpr::getGetElementPtr(Image->getValueType(), Image, ZeroZero);
    auto *ImageE =
        ConstantExpr::getGetElementPtr(Image->getValueType(), Image, ZeroSize);

    ImagesInits.push_back(ConstantStruct::get(getDeviceImageTy(M), ImageB,
                                              ImageE, EntriesB, EntriesE));
  }

  // Then create images array.
  auto *ImagesData = ConstantArray::get(
      ArrayType::get(getDeviceImageTy(M), ImagesInits.size()), ImagesInits);

  auto *Images =
      new GlobalVariable(M, ImagesData->getType(), /*isConstant*/ true,
                         GlobalValue::InternalLinkage, ImagesData,
                         ".omp_offloading.device_images");
  Images->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  auto *ImagesB =
      ConstantExpr::getGetElementPtr(Images->getValueType(), Images, ZeroZero);

  // And finally create the binary descriptor object.
  auto *DescInit = ConstantStruct::get(
      getBinDescTy(M),
      ConstantInt::get(Type::getInt32Ty(C), ImagesInits.size()), ImagesB,
      EntriesB, EntriesE);

  return new GlobalVariable(M, DescInit->getType(), /*isConstant*/ true,
                            GlobalValue::InternalLinkage, DescInit,
                            ".omp_offloading.descriptor");
}

void createRegisterFunction(Module &M, GlobalVariable *BinDesc) {
  LLVMContext &C = M.getContext();
  auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                ".omp_offloading.descriptor_reg", &M);
  Func->setSection(".text.startup");

  // Get __tgt_register_lib function declaration.
  auto *RegFuncTy = FunctionType::get(Type::getVoidTy(C), getBinDescPtrTy(M),
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

void createUnregisterFunction(Module &M, GlobalVariable *BinDesc) {
  LLVMContext &C = M.getContext();
  auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                ".omp_offloading.descriptor_unreg", &M);
  Func->setSection(".text.startup");

  // Get __tgt_unregister_lib function declaration.
  auto *UnRegFuncTy = FunctionType::get(Type::getVoidTy(C), getBinDescPtrTy(M),
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

// struct fatbin_wrapper {
//  int32_t magic;
//  int32_t version;
//  void *image;
//  void *reserved;
//};
StructType *getFatbinWrapperTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *FatbinTy = StructType::getTypeByName(C, "fatbin_wrapper");
  if (!FatbinTy)
    FatbinTy = StructType::create("fatbin_wrapper", Type::getInt32Ty(C),
                                  Type::getInt32Ty(C), Type::getInt8PtrTy(C),
                                  Type::getInt8PtrTy(C));
  return FatbinTy;
}

/// Embed the image \p Image into the module \p M so it can be found by the
/// runtime.
GlobalVariable *createFatbinDesc(Module &M, ArrayRef<char> Image) {
  LLVMContext &C = M.getContext();
  llvm::Type *Int8PtrTy = Type::getInt8PtrTy(C);
  llvm::Triple Triple = llvm::Triple(M.getTargetTriple());

  // Create the global string containing the fatbinary.
  StringRef FatbinConstantSection =
      Triple.isMacOSX() ? "__NV_CUDA,__nv_fatbin" : ".nv_fatbin";
  auto *Data = ConstantDataArray::get(C, Image);
  auto *Fatbin = new GlobalVariable(M, Data->getType(), /*isConstant*/ true,
                                    GlobalVariable::InternalLinkage, Data,
                                    ".fatbin_image");
  Fatbin->setSection(FatbinConstantSection);

  // Create the fatbinary wrapper
  StringRef FatbinWrapperSection =
      Triple.isMacOSX() ? "__NV_CUDA,__fatbin" : ".nvFatBinSegment";
  Constant *FatbinWrapper[] = {
      ConstantInt::get(Type::getInt32Ty(C), CudaFatMagic),
      ConstantInt::get(Type::getInt32Ty(C), 1),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Fatbin, Int8PtrTy),
      ConstantPointerNull::get(Type::getInt8PtrTy(C))};

  Constant *FatbinInitializer =
      ConstantStruct::get(getFatbinWrapperTy(M), FatbinWrapper);

  auto *FatbinDesc =
      new GlobalVariable(M, getFatbinWrapperTy(M),
                         /*isConstant*/ true, GlobalValue::InternalLinkage,
                         FatbinInitializer, ".fatbin_wrapper");
  FatbinDesc->setSection(FatbinWrapperSection);
  FatbinDesc->setAlignment(Align(8));

  // We create a dummy entry to ensure the linker will define the begin / end
  // symbols. The CUDA runtime should ignore the null address if we attempt to
  // register it.
  auto *DummyInit =
      ConstantAggregateZero::get(ArrayType::get(getEntryTy(M), 0u));
  auto *DummyEntry = new GlobalVariable(
      M, DummyInit->getType(), true, GlobalVariable::ExternalLinkage, DummyInit,
      "__dummy.cuda_offloading.entry");
  DummyEntry->setSection("cuda_offloading_entries");
  DummyEntry->setVisibility(GlobalValue::HiddenVisibility);

  return FatbinDesc;
}

/// Create the register globals function. We will iterate all of the offloading
/// entries stored at the begin / end symbols and register them according to
/// their type. This creates the following function in IR:
///
/// extern struct __tgt_offload_entry __start_cuda_offloading_entries;
/// extern struct __tgt_offload_entry __stop_cuda_offloading_entries;
///
/// extern void __cudaRegisterFunction(void **, void *, void *, void *, int,
///                                    void *, void *, void *, void *, int *);
/// extern void __cudaRegisterVar(void **, void *, void *, void *, int32_t,
///                               int64_t, int32_t, int32_t);
///
/// void __cudaRegisterTest(void **fatbinHandle) {
///   for (struct __tgt_offload_entry *entry = &__start_cuda_offloading_entries;
///        entry != &__stop_cuda_offloading_entries; ++entry) {
///     if (!entry->size)
///       __cudaRegisterFunction(fatbinHandle, entry->addr, entry->name,
///                              entry->name, -1, 0, 0, 0, 0, 0);
///     else
///       __cudaRegisterVar(fatbinHandle, entry->addr, entry->name, entry->name,
///                         0, entry->size, 0, 0);
///   }
/// }
///
/// TODO: This only registers functions are variables. Additional support is
///       required for texture / surface / managed variables.
Function *createRegisterGlobalsFunction(Module &M) {
  LLVMContext &C = M.getContext();
  // Get the __cudaRegisterFunction function declaration.
  auto *RegFuncTy = FunctionType::get(
      Type::getInt32Ty(C),
      {Type::getInt8PtrTy(C)->getPointerTo(), Type::getInt8PtrTy(C),
       Type::getInt8PtrTy(C), Type::getInt8PtrTy(C), Type::getInt32Ty(C),
       Type::getInt8PtrTy(C), Type::getInt8PtrTy(C), Type::getInt8PtrTy(C),
       Type::getInt8PtrTy(C), Type::getInt32PtrTy(C)},
      /*isVarArg*/ false);
  FunctionCallee RegFunc =
      M.getOrInsertFunction("__cudaRegisterFunction", RegFuncTy);

  // Get the __cudaRegisterVar function declaration.
  auto *RegVarTy = FunctionType::get(
      Type::getInt32Ty(C),
      {Type::getInt8PtrTy(C)->getPointerTo(), Type::getInt8PtrTy(C),
       Type::getInt8PtrTy(C), Type::getInt8PtrTy(C), Type::getInt32Ty(C),
       getSizeTTy(M), Type::getInt32Ty(C), Type::getInt32Ty(C)},
      /*isVarArg*/ false);
  FunctionCallee RegVar = M.getOrInsertFunction("__cudaRegisterVar", RegVarTy);

  // Create the references to the start / stop symbols defined by the linker.
  auto *EntriesB = new GlobalVariable(
      M, ArrayType::get(getEntryTy(M), 0), /*isConstant*/ true,
      GlobalValue::ExternalLinkage,
      /*Initializer*/ nullptr, "__start_cuda_offloading_entries");
  EntriesB->setVisibility(GlobalValue::HiddenVisibility);
  auto *EntriesE = new GlobalVariable(
      M, ArrayType::get(getEntryTy(M), 0), /*isConstant*/ true,
      GlobalValue::ExternalLinkage,
      /*Initializer*/ nullptr, "__stop_cuda_offloading_entries");
  EntriesE->setVisibility(GlobalValue::HiddenVisibility);

  auto *RegGlobalsTy = FunctionType::get(Type::getVoidTy(C),
                                         Type::getInt8PtrTy(C)->getPointerTo(),
                                         /*isVarArg*/ false);
  auto *RegGlobalsFn = Function::Create(
      RegGlobalsTy, GlobalValue::InternalLinkage, ".cuda.globals_reg", &M);
  RegGlobalsFn->setSection(".text.startup");

  // Create the loop to register all the entries.
  IRBuilder<> Builder(BasicBlock::Create(C, "entry", RegGlobalsFn));
  auto *EntryBB = BasicBlock::Create(C, "while.entry", RegGlobalsFn);
  auto *IfThenBB = BasicBlock::Create(C, "if.then", RegGlobalsFn);
  auto *IfElseBB = BasicBlock::Create(C, "if.else", RegGlobalsFn);
  auto *IfEndBB = BasicBlock::Create(C, "if.end", RegGlobalsFn);
  auto *ExitBB = BasicBlock::Create(C, "while.end", RegGlobalsFn);

  auto *EntryCmp = Builder.CreateICmpNE(EntriesB, EntriesE);
  Builder.CreateCondBr(EntryCmp, EntryBB, ExitBB);
  Builder.SetInsertPoint(EntryBB);
  auto *Entry = Builder.CreatePHI(getEntryPtrTy(M), 2, "entry");
  auto *AddrPtr =
      Builder.CreateInBoundsGEP(getEntryTy(M), Entry,
                                {ConstantInt::get(getSizeTTy(M), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 0)});
  auto *Addr = Builder.CreateLoad(Type::getInt8PtrTy(C), AddrPtr, "addr");
  auto *NamePtr =
      Builder.CreateInBoundsGEP(getEntryTy(M), Entry,
                                {ConstantInt::get(getSizeTTy(M), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 1)});
  auto *Name = Builder.CreateLoad(Type::getInt8PtrTy(C), NamePtr, "name");
  auto *SizePtr =
      Builder.CreateInBoundsGEP(getEntryTy(M), Entry,
                                {ConstantInt::get(getSizeTTy(M), 0),
                                 ConstantInt::get(Type::getInt32Ty(C), 2)});
  auto *Size = Builder.CreateLoad(getSizeTTy(M), SizePtr, "size");
  auto *FnCond =
      Builder.CreateICmpEQ(Size, ConstantInt::getNullValue(getSizeTTy(M)));
  Builder.CreateCondBr(FnCond, IfThenBB, IfElseBB);
  Builder.SetInsertPoint(IfThenBB);
  Builder.CreateCall(RegFunc,
                     {RegGlobalsFn->arg_begin(), Addr, Name, Name,
                      ConstantInt::get(Type::getInt32Ty(C), -1),
                      ConstantPointerNull::get(Type::getInt8PtrTy(C)),
                      ConstantPointerNull::get(Type::getInt8PtrTy(C)),
                      ConstantPointerNull::get(Type::getInt8PtrTy(C)),
                      ConstantPointerNull::get(Type::getInt8PtrTy(C)),
                      ConstantPointerNull::get(Type::getInt32PtrTy(C))});
  Builder.CreateBr(IfEndBB);
  Builder.SetInsertPoint(IfElseBB);
  Builder.CreateCall(RegVar, {RegGlobalsFn->arg_begin(), Addr, Name, Name,
                              ConstantInt::get(Type::getInt32Ty(C), 0), Size,
                              ConstantInt::get(Type::getInt32Ty(C), 0),
                              ConstantInt::get(Type::getInt32Ty(C), 0)});
  Builder.CreateBr(IfEndBB);
  Builder.SetInsertPoint(IfEndBB);
  auto *NewEntry = Builder.CreateInBoundsGEP(
      getEntryTy(M), Entry, ConstantInt::get(getSizeTTy(M), 1));
  auto *Cmp = Builder.CreateICmpEQ(
      NewEntry,
      ConstantExpr::getInBoundsGetElementPtr(
          ArrayType::get(getEntryTy(M), 0), EntriesE,
          ArrayRef<Constant *>({ConstantInt::get(getSizeTTy(M), 0),
                                ConstantInt::get(getSizeTTy(M), 0)})));
  Entry->addIncoming(
      ConstantExpr::getInBoundsGetElementPtr(
          ArrayType::get(getEntryTy(M), 0), EntriesB,
          ArrayRef<Constant *>({ConstantInt::get(getSizeTTy(M), 0),
                                ConstantInt::get(getSizeTTy(M), 0)})),
      &RegGlobalsFn->getEntryBlock());
  Entry->addIncoming(NewEntry, IfEndBB);
  Builder.CreateCondBr(Cmp, ExitBB, EntryBB);
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  return RegGlobalsFn;
}

// Create the constructor and destructor to register the fatbinary with the CUDA
// runtime.
void createRegisterFatbinFunction(Module &M, GlobalVariable *FatbinDesc) {
  LLVMContext &C = M.getContext();
  auto *CtorFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *CtorFunc = Function::Create(CtorFuncTy, GlobalValue::InternalLinkage,
                                    ".cuda.fatbin_reg", &M);
  CtorFunc->setSection(".text.startup");

  auto *DtorFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *DtorFunc = Function::Create(DtorFuncTy, GlobalValue::InternalLinkage,
                                    ".cuda.fatbin_unreg", &M);
  DtorFunc->setSection(".text.startup");

  // Get the __cudaRegisterFatBinary function declaration.
  auto *RegFatTy = FunctionType::get(Type::getInt8PtrTy(C)->getPointerTo(),
                                     Type::getInt8PtrTy(C),
                                     /*isVarArg*/ false);
  FunctionCallee RegFatbin =
      M.getOrInsertFunction("__cudaRegisterFatBinary", RegFatTy);
  // Get the __cudaRegisterFatBinaryEnd function declaration.
  auto *RegFatEndTy = FunctionType::get(Type::getVoidTy(C),
                                        Type::getInt8PtrTy(C)->getPointerTo(),
                                        /*isVarArg*/ false);
  FunctionCallee RegFatbinEnd =
      M.getOrInsertFunction("__cudaRegisterFatBinaryEnd", RegFatEndTy);
  // Get the __cudaUnregisterFatBinary function declaration.
  auto *UnregFatTy = FunctionType::get(Type::getVoidTy(C),
                                       Type::getInt8PtrTy(C)->getPointerTo(),
                                       /*isVarArg*/ false);
  FunctionCallee UnregFatbin =
      M.getOrInsertFunction("__cudaUnregisterFatBinary", UnregFatTy);

  auto *AtExitTy =
      FunctionType::get(Type::getInt32Ty(C), DtorFuncTy->getPointerTo(),
                        /*isVarArg*/ false);
  FunctionCallee AtExit = M.getOrInsertFunction("atexit", AtExitTy);

  auto *BinaryHandleGlobal = new llvm::GlobalVariable(
      M, Type::getInt8PtrTy(C)->getPointerTo(), false,
      llvm::GlobalValue::InternalLinkage,
      llvm::ConstantPointerNull::get(Type::getInt8PtrTy(C)->getPointerTo()),
      ".cuda.binary_handle");

  // Create the constructor to register this image with the runtime.
  IRBuilder<> CtorBuilder(BasicBlock::Create(C, "entry", CtorFunc));
  CallInst *Handle = CtorBuilder.CreateCall(
      RegFatbin, ConstantExpr::getPointerBitCastOrAddrSpaceCast(
                     FatbinDesc, Type::getInt8PtrTy(C)));
  CtorBuilder.CreateAlignedStore(
      Handle, BinaryHandleGlobal,
      Align(M.getDataLayout().getPointerTypeSize(Type::getInt8PtrTy(C))));
  CtorBuilder.CreateCall(createRegisterGlobalsFunction(M), Handle);
  CtorBuilder.CreateCall(RegFatbinEnd, Handle);
  CtorBuilder.CreateCall(AtExit, DtorFunc);
  CtorBuilder.CreateRetVoid();

  // Create the destructor to unregister the image with the runtime. We cannot
  // use a standard global destructor after CUDA 9.2 so this must be called by
  // `atexit()` intead.
  IRBuilder<> DtorBuilder(BasicBlock::Create(C, "entry", DtorFunc));
  LoadInst *BinaryHandle = DtorBuilder.CreateAlignedLoad(
      Type::getInt8PtrTy(C)->getPointerTo(), BinaryHandleGlobal,
      Align(M.getDataLayout().getPointerTypeSize(Type::getInt8PtrTy(C))));
  DtorBuilder.CreateCall(UnregFatbin, BinaryHandle);
  DtorBuilder.CreateRetVoid();

  // Add this function to constructors.
  appendToGlobalCtors(M, CtorFunc, /*Priority*/ 1);
}

} // namespace

Error wrapOpenMPBinaries(Module &M, ArrayRef<ArrayRef<char>> Images) {
  GlobalVariable *Desc = createBinDesc(M, Images);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No binary descriptors created.");
  createRegisterFunction(M, Desc);
  createUnregisterFunction(M, Desc);
  return Error::success();
}

Error wrapCudaBinary(Module &M, ArrayRef<char> Image) {
  GlobalVariable *Desc = createFatbinDesc(M, Image);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No fatinbary section created.");

  createRegisterFatbinFunction(M, Desc);
  return Error::success();
}
