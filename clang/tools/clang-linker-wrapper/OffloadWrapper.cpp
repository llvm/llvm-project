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
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace {

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

} // namespace

Error wrapBinaries(Module &M, ArrayRef<ArrayRef<char>> Images) {
  GlobalVariable *Desc = createBinDesc(M, Images);
  if (!Desc)
    return createStringError(inconvertibleErrorCode(),
                             "No binary descriptors created.");
  createRegisterFunction(M, Desc);
  createUnregisterFunction(M, Desc);
  return Error::success();
}
