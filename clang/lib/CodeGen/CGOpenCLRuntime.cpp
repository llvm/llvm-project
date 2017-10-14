//===----- CGOpenCLRuntime.cpp - Interface to OpenCL Runtimes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

  llvm::LLVMContext& Ctx = CGM.getLLVMContext();
  uint32_t AddrSpc = CGM.getContext().getTargetAddressSpace(
      CGM.getTarget().getOpenCLTypeAddrSpace(T));
  switch (cast<BuiltinType>(T)->getKind()) {
  default:
    llvm_unreachable("Unexpected opencl builtin type!");
    return nullptr;
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix) \
  case BuiltinType::Id: \
    return llvm::PointerType::get( \
        llvm::StructType::create(Ctx, "opencl." #ImgType "_" #Suffix "_t"), \
        AddrSpc);
#include "clang/Basic/OpenCLImageTypes.def"
  case BuiltinType::OCLSampler:
    return getSamplerType(T);
  case BuiltinType::OCLEvent:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.event_t"), AddrSpc);
  case BuiltinType::OCLClkEvent:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.clk_event_t"), AddrSpc);
  case BuiltinType::OCLQueue:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.queue_t"), AddrSpc);
  case BuiltinType::OCLReserveID:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.reserve_id_t"), AddrSpc);
  }
}

llvm::Type *CGOpenCLRuntime::getPipeType(const PipeType *T) {
  if (!PipeTy){
    uint32_t PipeAddrSpc = CGM.getContext().getTargetAddressSpace(
        CGM.getTarget().getOpenCLTypeAddrSpace(T));
    PipeTy = llvm::PointerType::get(llvm::StructType::create(
      CGM.getLLVMContext(), "opencl.pipe_t"), PipeAddrSpc);
  }

  return PipeTy;
}

llvm::PointerType *CGOpenCLRuntime::getSamplerType(const Type *T) {
  if (!SamplerTy)
    SamplerTy = llvm::PointerType::get(llvm::StructType::create(
      CGM.getLLVMContext(), "opencl.sampler_t"),
      CGM.getContext().getTargetAddressSpace(
          CGM.getTarget().getOpenCLTypeAddrSpace(T)));
  return SamplerTy;
}

llvm::Value *CGOpenCLRuntime::getPipeElemSize(const Expr *PipeArg) {
  const PipeType *PipeTy = PipeArg->getType()->getAs<PipeType>();
  // The type of the last (implicit) argument to be passed.
  llvm::Type *Int32Ty = llvm::IntegerType::getInt32Ty(CGM.getLLVMContext());
  unsigned TypeSize = CGM.getContext()
                          .getTypeSizeInChars(PipeTy->getElementType())
                          .getQuantity();
  return llvm::ConstantInt::get(Int32Ty, TypeSize, false);
}

llvm::Value *CGOpenCLRuntime::getPipeElemAlign(const Expr *PipeArg) {
  const PipeType *PipeTy = PipeArg->getType()->getAs<PipeType>();
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

CGOpenCLRuntime::EnqueuedBlockInfo
CGOpenCLRuntime::emitOpenCLEnqueuedBlock(CodeGenFunction &CGF, const Expr *E) {
  // The block literal may be assigned to a const variable. Chasing down
  // to get the block literal.
  if (auto DR = dyn_cast<DeclRefExpr>(E)) {
    E = cast<VarDecl>(DR->getDecl())->getInit();
  }
  if (auto Cast = dyn_cast<CastExpr>(E)) {
    E = Cast->getSubExpr();
  }
  auto *Block = cast<BlockExpr>(E);

  // The same block literal may be enqueued multiple times. Cache it if
  // possible.
  auto Loc = EnqueuedBlockMap.find(Block);
  if (Loc != EnqueuedBlockMap.end()) {
    return Loc->second;
  }

  // Emit block literal as a common block expression and get the block invoke
  // function.
  llvm::Function *Invoke;
  auto *V = CGF.EmitBlockLiteral(cast<BlockExpr>(Block), &Invoke);
  auto *F = CGF.getTargetHooks().createEnqueuedBlockKernel(
      CGF, Invoke, V->stripPointerCasts());

  // The common part of the post-processing of the kernel goes here.
  F->addFnAttr(llvm::Attribute::NoUnwind);
  F->setCallingConv(
      CGF.getTypes().ClangCallConvToLLVMCallConv(CallingConv::CC_OpenCLKernel));
  EnqueuedBlockInfo Info{F, V};
  EnqueuedBlockMap[Block] = Info;
  return Info;
}
