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
  uint32_t ImgAddrSpc =
    CGM.getContext().getTargetAddressSpace(LangAS::opencl_global);
  switch (cast<BuiltinType>(T)->getKind()) {
  default: 
    llvm_unreachable("Unexpected opencl builtin type!");
    return nullptr;
  case BuiltinType::OCLImage1d:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.image1d_t"), ImgAddrSpc);
  case BuiltinType::OCLImage1dArray:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.image1d_array_t"), ImgAddrSpc);
  case BuiltinType::OCLImage1dBuffer:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.image1d_buffer_t"), ImgAddrSpc);
  case BuiltinType::OCLImage2d:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.image2d_t"), ImgAddrSpc);
  case BuiltinType::OCLImage2dArray:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.image2d_array_t"), ImgAddrSpc);
  case BuiltinType::OCLImage2dDepth:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.image2d_depth_t"), ImgAddrSpc);
  case BuiltinType::OCLImage2dArrayDepth:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.image2d_array_depth_t"),
        ImgAddrSpc);
  case BuiltinType::OCLImage2dMSAA:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.image2d_msaa_t"), ImgAddrSpc);
  case BuiltinType::OCLImage2dArrayMSAA:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.image2d_array_msaa_t"),
        ImgAddrSpc);
  case BuiltinType::OCLImage2dMSAADepth:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.image2d_msaa_depth_t"),
        ImgAddrSpc);
  case BuiltinType::OCLImage2dArrayMSAADepth:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.image2d_array_msaa_depth_t"),
        ImgAddrSpc);
  case BuiltinType::OCLImage3d:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.image3d_t"), ImgAddrSpc);
  case BuiltinType::OCLSampler:
    return llvm::IntegerType::get(Ctx, 32);
  case BuiltinType::OCLEvent:
    return llvm::PointerType::get(llvm::StructType::create(
                           Ctx, "opencl.event_t"), 0);
  case BuiltinType::OCLClkEvent:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.clk_event_t"), 0);
  case BuiltinType::OCLQueue:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.queue_t"), 0);
  case BuiltinType::OCLNDRange:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.ndrange_t"), 0);
  case BuiltinType::OCLReserveID:
    return llvm::PointerType::get(
        llvm::StructType::create(Ctx, "opencl.reserve_id_t"), 0);
  }
}

llvm::Type *CGOpenCLRuntime::getPipeType() {
  if (!PipeTy){
    uint32_t PipeAddrSpc =
      CGM.getContext().getTargetAddressSpace(LangAS::opencl_global);
    PipeTy = llvm::PointerType::get(llvm::StructType::create(
      CGM.getLLVMContext(), "opencl.pipe_t"), PipeAddrSpc);
  }

  return PipeTy;
}
