//===---- AMDGPU.cpp - AMDGPU-specific CIR CodeGen ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides AMDGPU-specific CIR CodeGen logic for function attributes.
//
//===----------------------------------------------------------------------===//

#include "../CIRGenModule.h"
#include "../TargetInfo.h"

#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::CIRGen;

bool clang::CIRGen::requiresAMDGPUProtectedVisibility(
    const Decl *d, cir::VisibilityKind visibility) {
  if (visibility != cir::VisibilityKind::Hidden)
    return false;

  return !d->hasAttr<OMPDeclareTargetDeclAttr>() &&
         (d->hasAttr<DeviceKernelAttr>() ||
          (isa<FunctionDecl>(d) && d->hasAttr<CUDAGlobalAttr>()) ||
          (isa<VarDecl>(d) &&
           (d->hasAttr<CUDADeviceAttr>() || d->hasAttr<CUDAConstantAttr>() ||
            cast<VarDecl>(d)->getType()->isCUDADeviceBuiltinSurfaceType() ||
            cast<VarDecl>(d)->getType()->isCUDADeviceBuiltinTextureType())));
}

namespace {

/// Handle amdgpu-flat-work-group-size attribute.
static void
handleAMDGPUFlatWorkGroupSizeAttr(const FunctionDecl *fd, cir::FuncOp func,
                                  CIRGenModule &cgm, CIRGenBuilderTy &builder,
                                  bool isOpenCLKernel, bool isHIPKernel) {
  const auto *flatWGS = fd->getAttr<AMDGPUFlatWorkGroupSizeAttr>();
  const auto *reqdWGS =
      cgm.getLangOpts().OpenCL ? fd->getAttr<ReqdWorkGroupSizeAttr>() : nullptr;

  if (flatWGS || reqdWGS) {
    unsigned min = 0, max = 0;
    if (flatWGS) {
      min = flatWGS->getMin()
                ->EvaluateKnownConstInt(cgm.getASTContext())
                .getExtValue();
      max = flatWGS->getMax()
                ->EvaluateKnownConstInt(cgm.getASTContext())
                .getExtValue();
    }
    if (reqdWGS && min == 0 && max == 0) {
      min = max = reqdWGS->getXDim()
                      ->EvaluateKnownConstInt(cgm.getASTContext())
                      .getExtValue() *
                  reqdWGS->getYDim()
                      ->EvaluateKnownConstInt(cgm.getASTContext())
                      .getExtValue() *
                  reqdWGS->getZDim()
                      ->EvaluateKnownConstInt(cgm.getASTContext())
                      .getExtValue();
    }
    if (min != 0) {
      assert(min <= max && "Min must be less than or equal Max");
      std::string attrVal = llvm::utostr(min) + "," + llvm::utostr(max);
      func->setAttr("cir.amdgpu-flat-work-group-size",
                    builder.getStringAttr(attrVal));
    } else {
      assert(max == 0 && "Max must be zero");
    }
  } else if (isOpenCLKernel || isHIPKernel) {
    // By default, restrict the maximum size to a value specified by
    // --gpu-max-threads-per-block=n or its default value for HIP.
    const unsigned openCLDefaultMaxWorkGroupSize = 256;
    const unsigned defaultMaxWorkGroupSize =
        isOpenCLKernel ? openCLDefaultMaxWorkGroupSize
                       : cgm.getLangOpts().GPUMaxThreadsPerBlock;
    std::string attrVal =
        std::string("1,") + llvm::utostr(defaultMaxWorkGroupSize);
    func->setAttr("cir.amdgpu-flat-work-group-size",
                  builder.getStringAttr(attrVal));
  }
}

/// Handle amdgpu-waves-per-eu attribute.
static void handleAMDGPUWavesPerEUAttr(const FunctionDecl *fd, cir::FuncOp func,
                                       CIRGenModule &cgm,
                                       CIRGenBuilderTy &builder) {
  const auto *attr = fd->getAttr<AMDGPUWavesPerEUAttr>();
  if (!attr)
    return;
  unsigned min =
      attr->getMin()->EvaluateKnownConstInt(cgm.getASTContext()).getExtValue();
  unsigned max = attr->getMax()
                     ? attr->getMax()
                           ->EvaluateKnownConstInt(cgm.getASTContext())
                           .getExtValue()
                     : 0;

  if (min != 0) {
    assert((max == 0 || min <= max) && "Min must be less than or equal Max");
    std::string attrVal = llvm::utostr(min);
    if (max != 0)
      attrVal = attrVal + "," + llvm::utostr(max);
    func->setAttr("cir.amdgpu-waves-per-eu", builder.getStringAttr(attrVal));
  } else {
    assert(max == 0 && "Max must be zero");
  }
}

/// Handle amdgpu-num-sgpr attribute.
static void handleAMDGPUNumSGPRAttr(const FunctionDecl *fd, cir::FuncOp func,
                                    CIRGenModule &cgm,
                                    CIRGenBuilderTy &builder) {
  const auto *attr = fd->getAttr<AMDGPUNumSGPRAttr>();
  if (!attr)
    return;

  uint32_t numSGPR = attr->getNumSGPR();
  if (numSGPR != 0) {
    func->setAttr("cir.amdgpu-num-sgpr",
                  builder.getStringAttr(llvm::utostr(numSGPR)));
  }
}

/// Handle amdgpu-num-vgpr attribute.
static void handleAMDGPUNumVGPRAttr(const FunctionDecl *fd, cir::FuncOp func,
                                    CIRGenModule &cgm,
                                    CIRGenBuilderTy &builder) {
  const auto *attr = fd->getAttr<AMDGPUNumVGPRAttr>();
  if (!attr)
    return;

  uint32_t numVGPR = attr->getNumVGPR();
  if (numVGPR != 0) {
    func->setAttr("cir.amdgpu-num-vgpr",
                  builder.getStringAttr(llvm::utostr(numVGPR)));
  }
}

/// Handle amdgpu-max-num-workgroups attribute.
static void handleAMDGPUMaxNumWorkGroupsAttr(const FunctionDecl *fd,
                                             cir::FuncOp func,
                                             CIRGenModule &cgm,
                                             CIRGenBuilderTy &builder) {
  const auto *attr = fd->getAttr<AMDGPUMaxNumWorkGroupsAttr>();
  if (!attr)
    return;
  uint32_t x = attr->getMaxNumWorkGroupsX()
                   ->EvaluateKnownConstInt(cgm.getASTContext())
                   .getExtValue();
  uint32_t y = attr->getMaxNumWorkGroupsY()
                   ? attr->getMaxNumWorkGroupsY()
                         ->EvaluateKnownConstInt(cgm.getASTContext())
                         .getExtValue()
                   : 1;
  uint32_t z = attr->getMaxNumWorkGroupsZ()
                   ? attr->getMaxNumWorkGroupsZ()
                         ->EvaluateKnownConstInt(cgm.getASTContext())
                         .getExtValue()
                   : 1;

  llvm::SmallString<32> attrVal;
  llvm::raw_svector_ostream os(attrVal);
  os << x << ',' << y << ',' << z;
  func->setAttr("cir.amdgpu-max-num-workgroups",
                builder.getStringAttr(attrVal.str()));
}

/// Handle amdgpu-cluster-dims attribute.
static void handleAMDGPUClusterDimsAttr(const FunctionDecl *fd,
                                        cir::FuncOp func, CIRGenModule &cgm,
                                        CIRGenBuilderTy &builder,
                                        bool isOpenCLKernel) {

  if (const auto *attr = fd->getAttr<CUDAClusterDimsAttr>()) {
    auto getExprVal = [&](const Expr *e) {
      return e ? e->EvaluateKnownConstInt(cgm.getASTContext()).getExtValue()
               : 1;
    };
    unsigned x = getExprVal(attr->getX());
    unsigned y = getExprVal(attr->getY());
    unsigned z = getExprVal(attr->getZ());

    llvm::SmallString<32> attrVal;
    llvm::raw_svector_ostream os(attrVal);
    os << x << ',' << y << ',' << z;
    func->setAttr("cir.amdgpu-cluster-dims",
                  builder.getStringAttr(attrVal.str()));
  }

  const TargetInfo &targetInfo = cgm.getASTContext().getTargetInfo();
  if ((isOpenCLKernel &&
       targetInfo.hasFeatureEnabled(targetInfo.getTargetOpts().FeatureMap,
                                    "clusters")) ||
      fd->hasAttr<CUDANoClusterAttr>()) {
    func->setAttr("cir.amdgpu-cluster-dims", builder.getStringAttr("0,0,0"));
  }
}

/// Handle amdgpu-ieee attribute.
static void handleAMDGPUIEEEAttr(cir::FuncOp func, CIRGenModule &cgm,
                                 CIRGenBuilderTy &builder) {
  if (!cgm.getCodeGenOpts().EmitIEEENaNCompliantInsts)
    func->setAttr("cir.amdgpu-ieee", builder.getStringAttr("false"));
}

/// Handle amdgpu-expand-waitcnt-profiling attribute.
static void handleAMDGPUExpandWaitcntProfilingAttr(cir::FuncOp func,
                                                   CIRGenModule &cgm,
                                                   CIRGenBuilderTy &builder) {
  if (cgm.getCodeGenOpts().AMDGPUExpandWaitcntProfiling)
    func->setAttr("cir.amdgpu-expand-waitcnt-profiling",
                  builder.getStringAttr(""));
}

} // namespace

void clang::CIRGen::setAMDGPUTargetFunctionAttributes(const Decl *decl,
                                                      cir::FuncOp func,
                                                      CIRGenModule &cgm) {
  if (func.isDeclaration())
    return;

  CIRGenBuilderTy &builder = cgm.getBuilder();

  const auto *fd = dyn_cast_or_null<FunctionDecl>(decl);
  if (fd) {
    const bool isOpenCLKernel =
        cgm.getLangOpts().OpenCL && fd->hasAttr<DeviceKernelAttr>();
    const bool isHIPKernel =
        cgm.getLangOpts().HIP && fd->hasAttr<CUDAGlobalAttr>();

    // TODO(CIR): Set amdgpu_kernel calling convention for HIP kernels.

    handleAMDGPUFlatWorkGroupSizeAttr(fd, func, cgm, builder, isOpenCLKernel,
                                      isHIPKernel);
    handleAMDGPUWavesPerEUAttr(fd, func, cgm, builder);
    handleAMDGPUNumSGPRAttr(fd, func, cgm, builder);
    handleAMDGPUNumVGPRAttr(fd, func, cgm, builder);
    handleAMDGPUMaxNumWorkGroupsAttr(fd, func, cgm, builder);
    handleAMDGPUClusterDimsAttr(fd, func, cgm, builder, isOpenCLKernel);
  }
  handleAMDGPUIEEEAttr(func, cgm, builder);
  handleAMDGPUExpandWaitcntProfilingAttr(func, cgm, builder);
}
