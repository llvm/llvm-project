//===----- CGOpenCLRuntime.cpp - Interface to OpenCL Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for OpenCL code generation.  Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CGOpenCLRuntime.h"
#include "CodeGenFunction.h"
#include "TargetInfo.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include <assert.h>

using namespace clang;
using namespace CodeGen;

CGOpenCLRuntime::~CGOpenCLRuntime() {}

void CGOpenCLRuntime::EmitWorkGroupLocalVarDecl(CodeGenFunction &CGF,
                                                const VarDecl &D) {
  return CGF.EmitStaticVarDecl(D, llvm::GlobalValue::InternalLinkage);
}

llvm::Type *CGOpenCLRuntime::convertOpenCLSpecificType(const Type *T) {
  assert(T->isOpenCLSpecificType() &&
         "Not an OpenCL specific type!");

  switch (cast<BuiltinType>(T)->getKind()) {
  default:
    llvm_unreachable("Unexpected opencl builtin type!");
    return nullptr;
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:                                                        \
    return getPointerType(T, "opencl." #ImgType "_" #Suffix "_t");
#include "clang/Basic/OpenCLImageTypes.def"
  case BuiltinType::OCLSampler:
    return getSamplerType(T);
  case BuiltinType::OCLEvent:
    return getPointerType(T, "opencl.event_t");
  case BuiltinType::OCLClkEvent:
    return getPointerType(T, "opencl.clk_event_t");
  case BuiltinType::OCLQueue:
    return getPointerType(T, "opencl.queue_t");
  case BuiltinType::OCLReserveID:
    return getPointerType(T, "opencl.reserve_id_t");
#define EXT_OPAQUE_TYPE(ExtType, Id, Ext)                                      \
  case BuiltinType::Id:                                                        \
    return getPointerType(T, "opencl." #ExtType);
#include "clang/Basic/OpenCLExtensionTypes.def"
  }
}

llvm::PointerType *CGOpenCLRuntime::getPointerType(const Type *T,
                                                   StringRef Name) {
  auto I = CachedTys.find(Name);
  if (I != CachedTys.end())
    return I->second;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  uint32_t AddrSpc = CGM.getContext().getTargetAddressSpace(
      CGM.getContext().getOpenCLTypeAddrSpace(T));
  auto *PTy =
      llvm::PointerType::get(llvm::StructType::create(Ctx, Name), AddrSpc);
  CachedTys[Name] = PTy;
  return PTy;
}

llvm::Type *CGOpenCLRuntime::getPipeType(const PipeType *T) {
  if (T->isReadOnly())
    return getPipeType(T, "opencl.pipe_ro_t", PipeROTy);
  else
    return getPipeType(T, "opencl.pipe_wo_t", PipeWOTy);
}

llvm::Type *CGOpenCLRuntime::getPipeType(const PipeType *T, StringRef Name,
                                         llvm::Type *&PipeTy) {
  if (!PipeTy)
    PipeTy = llvm::PointerType::get(llvm::StructType::create(
      CGM.getLLVMContext(), Name),
      CGM.getContext().getTargetAddressSpace(
          CGM.getContext().getOpenCLTypeAddrSpace(T)));
  return PipeTy;
}

llvm::PointerType *CGOpenCLRuntime::getSamplerType(const Type *T) {
  if (!SamplerTy)
    SamplerTy = llvm::PointerType::get(llvm::StructType::create(
      CGM.getLLVMContext(), "opencl.sampler_t"),
      CGM.getContext().getTargetAddressSpace(
          CGM.getContext().getOpenCLTypeAddrSpace(T)));
  return SamplerTy;
}

llvm::Value *CGOpenCLRuntime::getPipeElemSize(const Expr *PipeArg) {
  const PipeType *PipeTy = PipeArg->getType()->castAs<PipeType>();
  // The type of the last (implicit) argument to be passed.
  llvm::Type *Int32Ty = llvm::IntegerType::getInt32Ty(CGM.getLLVMContext());
  unsigned TypeSize = CGM.getContext()
                          .getTypeSizeInChars(PipeTy->getElementType())
                          .getQuantity();
  return llvm::ConstantInt::get(Int32Ty, TypeSize, false);
}

llvm::Value *CGOpenCLRuntime::getPipeElemAlign(const Expr *PipeArg) {
  const PipeType *PipeTy = PipeArg->getType()->castAs<PipeType>();
  // The type of the last (implicit) argument to be passed.
  llvm::Type *Int32Ty = llvm::IntegerType::getInt32Ty(CGM.getLLVMContext());
  unsigned TypeSize = CGM.getContext()
                          .getTypeAlignInChars(PipeTy->getElementType())
                          .getQuantity();
  return llvm::ConstantInt::get(Int32Ty, TypeSize, false);
}

llvm::PointerType *CGOpenCLRuntime::getGenericVoidPointerType() {
  assert(CGM.getLangOpts().OpenCL);
  return llvm::IntegerType::getInt8PtrTy(
      CGM.getLLVMContext(),
      CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic));
}

// Get the block literal from an expression derived from the block expression.
// OpenCL v2.0 s6.12.5:
// Block variable declarations are implicitly qualified with const. Therefore
// all block variables must be initialized at declaration time and may not be
// reassigned.
static const BlockExpr *getBlockExpr(const Expr *E) {
  const Expr *Prev = nullptr; // to make sure we do not stuck in infinite loop.
  while(!isa<BlockExpr>(E) && E != Prev) {
    Prev = E;
    E = E->IgnoreCasts();
    if (auto DR = dyn_cast<DeclRefExpr>(E)) {
      E = cast<VarDecl>(DR->getDecl())->getInit();
    }
  }
  return cast<BlockExpr>(E);
}

/// Record emitted llvm invoke function and llvm block literal for the
/// corresponding block expression.
void CGOpenCLRuntime::recordBlockInfo(const BlockExpr *E,
                                      llvm::Function *InvokeF,
                                      llvm::Value *Block) {
  assert(EnqueuedBlockMap.find(E) == EnqueuedBlockMap.end() &&
         "Block expression emitted twice");
  assert(isa<llvm::Function>(InvokeF) && "Invalid invoke function");
  assert(Block->getType()->isPointerTy() && "Invalid block literal type");
  EnqueuedBlockMap[E].InvokeFunc = InvokeF;
  EnqueuedBlockMap[E].BlockArg = Block;
  EnqueuedBlockMap[E].Kernel = nullptr;
}

llvm::Function *CGOpenCLRuntime::getInvokeFunction(const Expr *E) {
  return EnqueuedBlockMap[getBlockExpr(E)].InvokeFunc;
}

CGOpenCLRuntime::EnqueuedBlockInfo
CGOpenCLRuntime::emitOpenCLEnqueuedBlock(CodeGenFunction &CGF, const Expr *E) {
  CGF.EmitScalarExpr(E);

  // The block literal may be assigned to a const variable. Chasing down
  // to get the block literal.
  const BlockExpr *Block = getBlockExpr(E);

  assert(EnqueuedBlockMap.find(Block) != EnqueuedBlockMap.end() &&
         "Block expression not emitted");

  // Do not emit the block wrapper again if it has been emitted.
  if (EnqueuedBlockMap[Block].Kernel) {
    return EnqueuedBlockMap[Block];
  }

  auto *F = CGF.getTargetHooks().createEnqueuedBlockKernel(
      CGF, EnqueuedBlockMap[Block].InvokeFunc,
      EnqueuedBlockMap[Block].BlockArg->stripPointerCasts());

  // The common part of the post-processing of the kernel goes here.
  F->addFnAttr(llvm::Attribute::NoUnwind);
  F->setCallingConv(
      CGF.getTypes().ClangCallConvToLLVMCallConv(CallingConv::CC_OpenCLKernel));
  EnqueuedBlockMap[Block].Kernel = F;
  return EnqueuedBlockMap[Block];
}
