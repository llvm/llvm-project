//===- CIRLowerContext.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/AST/ASTContext.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRLowerContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

CIRLowerContext::CIRLowerContext(mlir::ModuleOp module,
                                 clang::LangOptions langOpts,
                                 clang::CodeGenOptions codeGenOpts)
    : mlirContext(module.getContext()), langOpts(std::move(langOpts)),
      codeGenOpts(std::move(codeGenOpts)) {}

CIRLowerContext::~CIRLowerContext() {}

mlir::Type CIRLowerContext::initBuiltinType(clang::BuiltinType::Kind builtinKind) {
  mlir::Type ty;

  // NOTE(cir): Clang does more stuff here. Not sure if we need to do the same.
  cir_cconv_assert(!cir::MissingFeatures::qualifiedTypes());
  switch (builtinKind) {
  case clang::BuiltinType::Char_S:
    ty = IntType::get(getMLIRContext(), 8, true);
    break;
  default:
    cir_cconv_unreachable("NYI");
  }

  types.push_back(ty);
  return ty;
}

void CIRLowerContext::initBuiltinTypes(const clang::TargetInfo &target,
                                       const clang::TargetInfo *auxTarget) {
  cir_cconv_assert((!this->target || this->target == &target) &&
                   "incorrect target reinitialization");
  this->target = &target;
  this->auxTarget = auxTarget;

  // C99 6.2.5p3.
  if (langOpts.CharIsSigned)
    charTy = initBuiltinType(clang::BuiltinType::Char_S);
  else
    cir_cconv_unreachable("NYI");
}

} // namespace cir