//========- CIRGenCUDANV.cpp - Interface to NVIDIA CUDA Runtime -----=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for CUDA code generation targeting the NVIDIA CUDA
// runtime library.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCUDARuntime.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/IR/Operation.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/Cuda.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/Casting.h"

using namespace clang;
using namespace clang::CIRGen;

namespace {

class CIRGenNVCUDARuntime : public CIRGenCUDARuntime {
protected:
  StringRef prefix;

  // Map a device stub function to a symbol for identifying kernel in host
  // code. For CUDA, the symbol for identifying the kernel is the same as the
  // device stub function. For HIP, they are different.
  llvm::StringMap<mlir::Operation *> kernelHandles;

  // Map a kernel handle to the kernel stub.
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> kernelStubs;
  // Mangle context for device.
  std::unique_ptr<MangleContext> deviceMC;

private:
  void emitDeviceStubBodyNew(CIRGenFunction &cgf, cir::FuncOp fn,
                             FunctionArgList &args);
  mlir::Value prepareKernelArgs(CIRGenFunction &cgf, mlir::Location loc,
                                FunctionArgList &args);
  mlir::Operation *getKernelHandle(cir::FuncOp fn, GlobalDecl gd) override;
  std::string addPrefixToName(StringRef funcName) const;
  std::string addUnderscoredPrefixToName(StringRef funcName) const;

public:
  CIRGenNVCUDARuntime(CIRGenModule &cgm);
  ~CIRGenNVCUDARuntime();

  void emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                      FunctionArgList &args) override;
};

} // namespace

std::string CIRGenNVCUDARuntime::addPrefixToName(StringRef funcName) const {
  return (prefix + funcName).str();
}

std::string
CIRGenNVCUDARuntime::addUnderscoredPrefixToName(StringRef funcName) const {
  return ("__" + prefix + funcName).str();
}

CIRGenNVCUDARuntime::CIRGenNVCUDARuntime(CIRGenModule &cgm)
    : CIRGenCUDARuntime(cgm),
      deviceMC(cgm.getASTContext().cudaNVInitDeviceMC()) {
  if (cgm.getLangOpts().OffloadViaLLVM)
    cgm.errorNYI("CIRGenNVCUDARuntime: Offload via LLVM");
  else if (cgm.getLangOpts().HIP)
    prefix = "hip";
  else
    prefix = "cuda";
}

mlir::Value CIRGenNVCUDARuntime::prepareKernelArgs(CIRGenFunction &cgf,
                                                   mlir::Location loc,
                                                   FunctionArgList &args) {
  CIRGenBuilderTy &builder = cgm.getBuilder();

  // Build void *args[] and populate with the addresses of kernel arguments.
  auto voidPtrArrayTy = cir::ArrayType::get(cgm.voidPtrTy, args.size());
  mlir::Value kernelArgs = builder.createAlloca(
      loc, cir::PointerType::get(voidPtrArrayTy), voidPtrArrayTy, "kernel_args",
      CharUnits::fromQuantity(16));

  mlir::Value kernelArgsDecayed =
      builder.createCast(cir::CastKind::array_to_ptrdecay, kernelArgs,
                         cir::PointerType::get(cgm.voidPtrTy));

  for (const auto &[i, arg] : llvm::enumerate(args)) {
    mlir::Value index =
        builder.getConstInt(loc, llvm::APInt(/*numBits=*/32, i));
    mlir::Value storePos =
        builder.createPtrStride(loc, kernelArgsDecayed, index);
    mlir::Value argAddr = cgf.getAddrOfLocalVar(arg).getPointer();
    mlir::Value argAsVoid = builder.createBitcast(argAddr, cgm.voidPtrTy);

    builder.CIRBaseBuilderTy::createStore(loc, argAsVoid, storePos);
  }

  return kernelArgsDecayed;
}

// CUDA 9.0+ uses new way to launch kernels. Parameters are packed in a local
// array and kernels are launched using cudaLaunchKernel().
void CIRGenNVCUDARuntime::emitDeviceStubBodyNew(CIRGenFunction &cgf,
                                                cir::FuncOp fn,
                                                FunctionArgList &args) {

  // This requires arguments to be sent to kernels in a different way.
  if (cgm.getLangOpts().OffloadViaLLVM)
    cgm.errorNYI("CIRGenNVCUDARuntime: Offload via LLVM");

  if (cgm.getLangOpts().HIP)
    cgm.errorNYI("CIRGenNVCUDARuntime: HIP Support");

  CIRGenBuilderTy &builder = cgm.getBuilder();
  mlir::Location loc = fn.getLoc();

  // For [cuda|hip]LaunchKernel, we must add another layer of indirection
  // to arguments. For example, for function `add(int a, float b)`,
  // we need to pass it as `void *args[2] = { &a, &b }`.
  mlir::Value kernelArgs = prepareKernelArgs(cgf, loc, args);

  // Lookup cudaLaunchKernel/hipLaunchKernel function.
  // HIP kernel launching API name depends on -fgpu-default-stream option. For
  // the default value 'legacy', it is hipLaunchKernel. For 'per-thread',
  // it is hipLaunchKernel_spt.
  // cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
  //                              void **args, size_t sharedMem,
  //                              cudaStream_t stream);
  // hipError_t hipLaunchKernel[_spt](const void *func, dim3 gridDim,
  //                                  dim3 blockDim, void **args,
  //                                  size_t sharedMem, hipStream_t stream);
  TranslationUnitDecl *tuDecl = cgm.getASTContext().getTranslationUnitDecl();
  DeclContext *dc = TranslationUnitDecl::castToDeclContext(tuDecl);

  // The default stream is usually stream 0 (the legacy default stream).
  // For per-thread default stream, we need a different LaunchKernel function.
  StringRef kernelLaunchAPI = "LaunchKernel";
  if (cgm.getLangOpts().GPUDefaultStream ==
      LangOptions::GPUDefaultStreamKind::PerThread)
    cgm.errorNYI("CUDA/HIP Stream per thread");

  std::string launchKernelName = addPrefixToName(kernelLaunchAPI);
  const IdentifierInfo &launchII =
      cgm.getASTContext().Idents.get(launchKernelName);
  FunctionDecl *cudaLaunchKernelFD = nullptr;
  for (NamedDecl *result : dc->lookup(&launchII)) {
    if (FunctionDecl *fd = dyn_cast<FunctionDecl>(result))
      cudaLaunchKernelFD = fd;
  }

  if (cudaLaunchKernelFD == nullptr) {
    cgm.error(cgf.curFuncDecl->getLocation(),
              "Can't find declaration for " + launchKernelName);
    return;
  }

  // Use this function to retrieve arguments for cudaLaunchKernel:
  // int __[cuda|hip]PopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t
  //                                *sharedMem, cudaStream_t *stream)
  //
  // Here [cuda|hip]Stream_t, while also being the 6th argument of
  // [cuda|hip]LaunchKernel, is a pointer to some opaque struct.

  mlir::Type dim3Ty = cgf.getTypes().convertType(
      cudaLaunchKernelFD->getParamDecl(1)->getType());
  mlir::Type streamTy = cgf.getTypes().convertType(
      cudaLaunchKernelFD->getParamDecl(5)->getType());

  mlir::Value gridDim =
      builder.createAlloca(loc, cir::PointerType::get(dim3Ty), dim3Ty,
                           "grid_dim", CharUnits::fromQuantity(8));
  mlir::Value blockDim =
      builder.createAlloca(loc, cir::PointerType::get(dim3Ty), dim3Ty,
                           "block_dim", CharUnits::fromQuantity(8));
  mlir::Value sharedMem =
      builder.createAlloca(loc, cir::PointerType::get(cgm.sizeTy), cgm.sizeTy,
                           "shared_mem", cgm.getSizeAlign());
  mlir::Value stream =
      builder.createAlloca(loc, cir::PointerType::get(streamTy), streamTy,
                           "stream", cgm.getPointerAlign());

  cir::FuncOp popConfig = cgm.createRuntimeFunction(
      cir::FuncType::get({gridDim.getType(), blockDim.getType(),
                          sharedMem.getType(), stream.getType()},
                         cgm.sInt32Ty),
      addUnderscoredPrefixToName("PopCallConfiguration"));
  cgf.emitRuntimeCall(loc, popConfig, {gridDim, blockDim, sharedMem, stream});

  // Now emit the call to cudaLaunchKernel
  // [cuda|hip]Error_t [cuda|hip]LaunchKernel(const void *func, dim3 gridDim,
  // dim3 blockDim,
  //                              void **args, size_t sharedMem,
  //                              [cuda|hip]Stream_t stream);

  // We now either pick the function or the stub global for cuda, hip
  // respectively.
  mlir::Value kernel = [&]() -> mlir::Value {
    if (cir::GlobalOp globalOp = llvm::dyn_cast_or_null<cir::GlobalOp>(
            kernelHandles[fn.getSymName()])) {
      cir::PointerType kernelTy = cir::PointerType::get(globalOp.getSymType());
      mlir::Value kernelVal = cir::GetGlobalOp::create(builder, loc, kernelTy,
                                                       globalOp.getSymName());
      return kernelVal;
    }
    if (cir::FuncOp funcOp = llvm::dyn_cast_or_null<cir::FuncOp>(
            kernelHandles[fn.getSymName()])) {
      cir::PointerType kernelTy =
          cir::PointerType::get(funcOp.getFunctionType());
      mlir::Value kernelVal =
          cir::GetGlobalOp::create(builder, loc, kernelTy, funcOp.getSymName());
      mlir::Value func = builder.createBitcast(kernelVal, cgm.voidPtrTy);
      return func;
    }
    llvm_unreachable("Expected stub handle to be cir::GlobalOp or FuncOp");
  }();

  CallArgList launchArgs;
  launchArgs.add(RValue::get(kernel),
                 cudaLaunchKernelFD->getParamDecl(0)->getType());
  launchArgs.add(
      RValue::getAggregate(Address(gridDim, CharUnits::fromQuantity(8))),
      cudaLaunchKernelFD->getParamDecl(1)->getType());
  launchArgs.add(
      RValue::getAggregate(Address(blockDim, CharUnits::fromQuantity(8))),
      cudaLaunchKernelFD->getParamDecl(2)->getType());
  launchArgs.add(RValue::get(kernelArgs),
                 cudaLaunchKernelFD->getParamDecl(3)->getType());
  launchArgs.add(
      RValue::get(builder.CIRBaseBuilderTy::createLoad(loc, sharedMem)),
      cudaLaunchKernelFD->getParamDecl(4)->getType());
  launchArgs.add(RValue::get(builder.CIRBaseBuilderTy::createLoad(loc, stream)),
                 cudaLaunchKernelFD->getParamDecl(5)->getType());

  mlir::Type launchTy =
      cgm.getTypes().convertType(cudaLaunchKernelFD->getType());
  mlir::Operation *cudaKernelLauncherFn = cgm.createRuntimeFunction(
      cast<cir::FuncType>(launchTy), launchKernelName);
  const CIRGenFunctionInfo &callInfo =
      cgm.getTypes().arrangeFunctionDeclaration(cudaLaunchKernelFD);
  cgf.emitCall(callInfo, CIRGenCallee::forDirect(cudaKernelLauncherFn),
               ReturnValueSlot(), launchArgs);

  if (cgm.getASTContext().getTargetInfo().getCXXABI().isMicrosoft() &&
      !cgf.getLangOpts().HIP)
    cgm.errorNYI("MSVC CUDA stub handling");
}

void CIRGenNVCUDARuntime::emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                                         FunctionArgList &args) {

  if (auto globalOp =
          llvm::dyn_cast<cir::GlobalOp>(kernelHandles[fn.getSymName()])) {
    CIRGenBuilderTy &builder = cgm.getBuilder();
    mlir::Type fnPtrTy = globalOp.getSymType();
    auto sym = mlir::FlatSymbolRefAttr::get(fn.getSymNameAttr());
    auto gv = cir::GlobalViewAttr::get(fnPtrTy, sym);

    globalOp->setAttr("initial_value", gv);
    globalOp->removeAttr("sym_visibility");
    globalOp->setAttr("alignment", builder.getI64IntegerAttr(
                                       cgm.getPointerAlign().getQuantity()));
  }

  // CUDA 9.0 changed the way to launch kernels.
  if (CudaFeatureEnabled(cgm.getTarget().getSDKVersion(),
                         CudaFeature::CUDA_USES_NEW_LAUNCH) ||
      (cgm.getLangOpts().HIP && cgm.getLangOpts().HIPUseNewLaunchAPI) ||
      cgm.getLangOpts().OffloadViaLLVM)
    emitDeviceStubBodyNew(cgf, fn, args);
  else
    cgm.errorNYI("Emit Stub Body Legacy");
}

CIRGenCUDARuntime *clang::CIRGen::createNVCUDARuntime(CIRGenModule &cgm) {
  return new CIRGenNVCUDARuntime(cgm);
}

CIRGenNVCUDARuntime::~CIRGenNVCUDARuntime() {}

mlir::Operation *CIRGenNVCUDARuntime::getKernelHandle(cir::FuncOp fn,
                                                      GlobalDecl gd) {

  // Check if we already have a kernel handle for this function
  auto it = kernelHandles.find(fn.getSymName());
  if (it != kernelHandles.end()) {
    mlir::Operation *oldHandle = it->second;
    // Here we know that the fn did not change. Return it
    if (kernelStubs[oldHandle] == fn)
      return oldHandle;

    // We've found the function name, but F itself has changed, so we need to
    // update the references.
    if (cgm.getLangOpts().HIP) {
      // For HIP compilation the handle itself does not change, so we only need
      // to update the Stub value.
      kernelStubs[oldHandle] = fn;
      return oldHandle;
    }
    // For non-HIP compilation, erase the old Stub and fall-through to creating
    // new entries.
    kernelStubs.erase(oldHandle);
  }

  // If not targeting HIP, store the function itself
  if (!cgm.getLangOpts().HIP) {
    kernelHandles[fn.getSymName()] = fn;
    kernelStubs[fn] = fn;
    return fn;
  }

  // Create a new CIR global variable to represent the kernel handle
  CIRGenBuilderTy &builder = cgm.getBuilder();
  StringRef globalName = cgm.getMangledName(
      gd.getWithKernelReferenceKind(KernelReferenceKind::Kernel));
  const VarDecl *varDecl = llvm::dyn_cast_or_null<VarDecl>(gd.getDecl());
  cir::GlobalOp globalOp =
      cgm.getOrCreateCIRGlobal(globalName, fn.getFunctionType().getReturnType(),
                               LangAS::Default, varDecl, NotForDefinition);

  globalOp->setAttr("alignment", builder.getI64IntegerAttr(
                                     cgm.getPointerAlign().getQuantity()));

  // Store references
  kernelHandles[fn.getSymName()] = globalOp;
  kernelStubs[globalOp] = fn;

  return globalOp;
}
