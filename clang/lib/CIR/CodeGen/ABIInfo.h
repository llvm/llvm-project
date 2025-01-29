//===----- ABIInfo.h - ABI information access & encapsulation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_ABIINFO_H
#define LLVM_CLANG_LIB_CIR_ABIINFO_H

#include "clang/AST/Type.h"
#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace clang::CIRGen {

class CIRGenCXXABI;
class CIRGenFunctionInfo;
class CIRGenTypes;

/// ABIInfo - Target specific hooks for defining how a type should be passed or
/// returned from functions.
class ABIInfo {
  ABIInfo() = delete;

public:
  CIRGenTypes &CGT;

  ABIInfo(CIRGenTypes &cgt) : CGT{cgt} {}

  virtual ~ABIInfo();

  CIRGenCXXABI &getCXXABI() const;
  clang::ASTContext &getContext() const;

  virtual void computeInfo(CIRGenFunctionInfo &FI) const = 0;

  virtual bool allowBFloatArgsAndRet() const { return false; }

  // Implement the Type::IsPromotableIntegerType for ABI specific needs. The
  // only difference is that this consideres bit-precise integer types as well.
  bool isPromotableIntegerTypeForABI(clang::QualType Ty) const;

  /// Returns the optimal vector memory type based on the given vector type. For
  /// example, on certain targets, a vector with 3 elements might be promoted to
  /// one with 4 elements to improve performance.
  virtual cir::VectorType
  getOptimalVectorMemoryType(cir::VectorType T,
                             const clang::LangOptions &Opt) const;
};

} // namespace clang::CIRGen

#endif
