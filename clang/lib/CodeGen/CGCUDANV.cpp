//===----- CGCUDANV.cpp - Interface to NVIDIA CUDA Runtime ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for CUDA code generation targeting the NVIDIA CUDA
// runtime library.
//
//===----------------------------------------------------------------------===//

#include "CGCUDARuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Decl.h"
#include "llvm/BasicBlock.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/CallSite.h"

#include <vector>

using namespace clang;
using namespace CodeGen;

namespace {

class CGNVCUDARuntime : public CGCUDARuntime {

private:
  llvm::Type *IntTy, *SizeTy;
  llvm::PointerType *CharPtrTy, *VoidPtrTy;

  llvm::Constant *getSetupArgumentFn() const;
  llvm::Constant *getLaunchFn() const;

public:
  CGNVCUDARuntime(CodeGenModule &CGM);

  void EmitDeviceStubBody(CodeGenFunction &CGF, FunctionArgList &Args);
};

}

CGNVCUDARuntime::CGNVCUDARuntime(CodeGenModule &CGM) : CGCUDARuntime(CGM) {
  CodeGen::CodeGenTypes &Types = CGM.getTypes();
  ASTContext &Ctx = CGM.getContext();

  IntTy = Types.ConvertType(Ctx.IntTy);
  SizeTy = Types.ConvertType(Ctx.getSizeType());

  CharPtrTy = llvm::PointerType::getUnqual(Types.ConvertType(Ctx.CharTy));
  VoidPtrTy = cast<llvm::PointerType>(Types.ConvertType(Ctx.VoidPtrTy));
}

llvm::Constant *CGNVCUDARuntime::getSetupArgumentFn() const {
  // cudaError_t cudaSetupArgument(void *, size_t, size_t)
  std::vector<llvm::Type*> Params;
  Params.push_back(VoidPtrTy);
  Params.push_back(SizeTy);
  Params.push_back(SizeTy);
  return CGM.CreateRuntimeFunction(llvm::FunctionType::get(IntTy,
                                                           Params, false),
                                   "cudaSetupArgument");
}

llvm::Constant *CGNVCUDARuntime::getLaunchFn() const {
  // cudaError_t cudaLaunch(char *)
  std::vector<llvm::Type*> Params;
  Params.push_back(CharPtrTy);
  return CGM.CreateRuntimeFunction(llvm::FunctionType::get(IntTy,
                                                           Params, false),
                                   "cudaLaunch");
}

void CGNVCUDARuntime::EmitDeviceStubBody(CodeGenFunction &CGF,
                                         FunctionArgList &Args) {
  // Build the argument value list and the argument stack struct type.
  llvm::SmallVector<llvm::Value *, 16> ArgValues;
  std::vector<llvm::Type *> ArgTypes;
  for (FunctionArgList::const_iterator I = Args.begin(), E = Args.end();
       I != E; ++I) {
    llvm::Value *V = CGF.GetAddrOfLocalVar(*I);
    ArgValues.push_back(V);
    assert(isa<llvm::PointerType>(V->getType()) && "Arg type not PointerType");
    ArgTypes.push_back(cast<llvm::PointerType>(V->getType())->getElementType());
  }
  llvm::StructType *ArgStackTy = llvm::StructType::get(
      CGF.getLLVMContext(), ArgTypes);

  llvm::BasicBlock *EndBlock = CGF.createBasicBlock("setup.end");

  // Emit the calls to cudaSetupArgument
  llvm::Constant *cudaSetupArgFn = getSetupArgumentFn();
  for (unsigned I = 0, E = Args.size(); I != E; ++I) {
    llvm::Value *Args[3];
    llvm::BasicBlock *NextBlock = CGF.createBasicBlock("setup.next");
    Args[0] = CGF.Builder.CreatePointerCast(ArgValues[I], VoidPtrTy);
    Args[1] = CGF.Builder.CreateIntCast(
        llvm::ConstantExpr::getSizeOf(ArgTypes[I]),
        SizeTy, false);
    Args[2] = CGF.Builder.CreateIntCast(
        llvm::ConstantExpr::getOffsetOf(ArgStackTy, I),
        SizeTy, false);
    llvm::CallSite CS = CGF.EmitCallOrInvoke(cudaSetupArgFn, Args);
    llvm::Constant *Zero = llvm::ConstantInt::get(IntTy, 0);
    llvm::Value *CSZero = CGF.Builder.CreateICmpEQ(CS.getInstruction(), Zero);
    CGF.Builder.CreateCondBr(CSZero, NextBlock, EndBlock);
    CGF.EmitBlock(NextBlock);
  }

  // Emit the call to cudaLaunch
  llvm::Constant *cudaLaunchFn = getLaunchFn();
  llvm::Value *Arg = CGF.Builder.CreatePointerCast(CGF.CurFn, CharPtrTy);
  CGF.EmitCallOrInvoke(cudaLaunchFn, Arg);
  CGF.EmitBranch(EndBlock);

  CGF.EmitBlock(EndBlock);
}

CGCUDARuntime *CodeGen::CreateNVCUDARuntime(CodeGenModule &CGM) {
  return new CGNVCUDARuntime(CGM);
}
