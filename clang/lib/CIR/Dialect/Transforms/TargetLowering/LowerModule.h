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
  const std::unique_ptr<clang::TargetInfo> target;
  std::unique_ptr<TargetLoweringInfo> targetLoweringInfo;
  std::unique_ptr<CIRCXXABI> abi;
  [[maybe_unused]] mlir::PatternRewriter &rewriter;

public:
  LowerModule(clang::LangOptions langOpts, clang::CodeGenOptions codeGenOpts,
              mlir::ModuleOp &module, std::unique_ptr<clang::TargetInfo> target,
              mlir::PatternRewriter &rewriter);
  ~LowerModule() = default;

  clang::TargetCXXABI::Kind getCXXABIKind() const {
    assert(!cir::MissingFeatures::lowerModuleLangOpts());
    return target->getCXXABI().getKind();
  }

  CIRCXXABI &getCXXABI() const { return *abi; }
  const clang::TargetInfo &getTarget() const { return *target; }
  mlir::MLIRContext *getMLIRContext() { return module.getContext(); }

  const TargetLoweringInfo &getTargetLoweringInfo();
};

std::unique_ptr<LowerModule> createLowerModule(mlir::ModuleOp module,
                                               mlir::PatternRewriter &rewriter);

} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERMODULE_H
