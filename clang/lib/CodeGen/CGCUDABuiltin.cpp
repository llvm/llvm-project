//===----- CGCUDABuiltin.cpp - Codegen for CUDA builtins ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Generates code for built-in CUDA calls which are not runtime-specific.
// (Runtime-specific codegen lives in CGCUDARuntime.)
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;
using namespace CodeGen;

static llvm::Function *GetVprintfDeclaration(llvm::Module &M) {
  llvm::Type *ArgTypes[] = {llvm::Type::getInt8PtrTy(M.getContext()),
                            llvm::Type::getInt8PtrTy(M.getContext())};
  llvm::FunctionType *VprintfFuncType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(M.getContext()), ArgTypes, false);

  if (auto* F = M.getFunction("vprintf")) {
    // Our CUDA system header declares vprintf with the right signature, so
    // nobody else should have been able to declare vprintf with a bogus
    // signature.
    assert(F->getFunctionType() == VprintfFuncType);
    return F;
  }

  // vprintf doesn't already exist; create a declaration and insert it into the
  // module.
  return llvm::Function::Create(
      VprintfFuncType, llvm::GlobalVariable::ExternalLinkage, "vprintf", &M);
}

// Transforms a call to printf into a call to the NVPTX vprintf syscall (which
// isn't particularly special; it's invoked just like a regular function).
// vprintf takes two args: A format string, and a pointer to a buffer containing
// the varargs.
//
// For example, the call
//
//   printf("format string", arg1, arg2, arg3);
//
// is converted into something resembling
//
//   char* buf = alloca(...);
//   *reinterpret_cast<Arg1*>(buf) = arg1;
//   *reinterpret_cast<Arg2*>(buf + ...) = arg2;
//   *reinterpret_cast<Arg3*>(buf + ...) = arg3;
//   vprintf("format string", buf);
//
// buf is aligned to the max of {alignof(Arg1), ...}.  Furthermore, each of the
// args is itself aligned to its preferred alignment.
//
// Note that by the time this function runs, E's args have already undergone the
// standard C vararg promotion (short -> int, float -> double, etc.).
RValue
CodeGenFunction::EmitCUDADevicePrintfCallExpr(const CallExpr *E,
                                              ReturnValueSlot ReturnValue) {
  assert(getLangOpts().CUDA);
  assert(getLangOpts().CUDAIsDevice);
  assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  const llvm::DataLayout &DL = CGM.getDataLayout();
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  CallArgList Args;
  EmitCallArgs(Args,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  // Figure out how large of a buffer we need to hold our varargs and how
  // aligned the buffer needs to be.  We start iterating at Arg[1], because
  // that's our first vararg.
  unsigned BufSize = 0;
  unsigned BufAlign = 0;
  for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
    const RValue& RV = Args[I].RV;
    llvm::Type* Ty = RV.getScalarVal()->getType();

    auto Align = DL.getPrefTypeAlignment(Ty);
    BufAlign = std::max(BufAlign, Align);
    // Add padding required to keep the current arg aligned.
    BufSize = llvm::alignTo(BufSize, Align);
    BufSize += DL.getTypeAllocSize(Ty);
  }

  // Construct and fill the buffer.
  llvm::Value* BufferPtr = nullptr;
  if (BufSize == 0) {
    // If there are no args, pass a null pointer to vprintf.
    BufferPtr = llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(Ctx));
  } else {
    BufferPtr = Builder.Insert(new llvm::AllocaInst(
        llvm::Type::getInt8Ty(Ctx), llvm::ConstantInt::get(Int32Ty, BufSize),
        BufAlign, "printf_arg_buf"));

    unsigned Offset = 0;
    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
      llvm::Value *Arg = Args[I].RV.getScalarVal();
      llvm::Type *Ty = Arg->getType();
      auto Align = DL.getPrefTypeAlignment(Ty);

      // Pad the buffer to Arg's alignment.
      Offset = llvm::alignTo(Offset, Align);

      // Store Arg into the buffer at Offset.
      llvm::Value *GEP =
          Builder.CreateGEP(BufferPtr, llvm::ConstantInt::get(Int32Ty, Offset));
      llvm::Value *Cast = Builder.CreateBitCast(GEP, Ty->getPointerTo());
      Builder.CreateAlignedStore(Arg, Cast, Align);
      Offset += DL.getTypeAllocSize(Ty);
    }
  }

  // Invoke vprintf and return.
  llvm::Function* VprintfFunc = GetVprintfDeclaration(CGM.getModule());
  return RValue::get(
      Builder.CreateCall(VprintfFunc, {Args[0].RV.getScalarVal(), BufferPtr}));
}
