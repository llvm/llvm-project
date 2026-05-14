//===--- LowerModule.h - Abstracts CIR's module lowering --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenModule.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H

#include "CIRCXXABI.h"
#include "TargetLoweringInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"
#include <memory>

namespace cir {

class LowerModule {
  mlir::ModuleOp module;
  clang::LangOptions langOpts;
  clang::CodeGenOptions codeGenOpts;
  const std::unique_ptr<clang::TargetInfo> target;
  std::unique_ptr<TargetLoweringInfo> targetLoweringInfo;
  std::unique_ptr<CIRCXXABI> abi;

public:
  LowerModule(clang::LangOptions langOpts, clang::CodeGenOptions codeGenOpts,
              mlir::ModuleOp &module,
              std::unique_ptr<clang::TargetInfo> target);
  ~LowerModule() = default;

  clang::TargetCXXABI::Kind getCXXABIKind() const {
    return target->getCXXABI().getKind();
  }

  CIRCXXABI &getCXXABI() const { return *abi; }
  const clang::TargetInfo &getTarget() const { return *target; }
  const clang::LangOptions &getLangOpts() const { return langOpts; }
  const clang::CodeGenOptions &getCodeGenOpts() const { return codeGenOpts; }
  mlir::MLIRContext *getMLIRContext() { return module.getContext(); }

  const TargetLoweringInfo &getTargetLoweringInfo();
};

/// Build a LowerModule from a parsed CIR module alone, deriving the target
/// from the `cir.triple` attribute. LangOptions and CodeGenOptions are
/// default-constructed; only the optimization level is recovered from the
/// `cir.opt_info` attribute when present.
std::unique_ptr<LowerModule> createLowerModule(mlir::ModuleOp module);

/// Build a LowerModule using the LangOptions, CodeGenOptions, and TargetInfo
/// from the surrounding compiler invocation. Preferred over the
/// module-only factory whenever a live invocation is available, since it
/// captures cc1 flags that influence lowering decisions (HIP/CUDA, OpenMP,
/// thread-safe statics, target features, etc.).
std::unique_ptr<LowerModule>
createLowerModule(mlir::ModuleOp module, const clang::LangOptions &langOpts,
                  const clang::CodeGenOptions &codeGenOpts,
                  std::unique_ptr<clang::TargetInfo> target);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
