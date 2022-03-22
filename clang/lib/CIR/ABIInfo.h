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

namespace cir {

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

  // Implement the Type::IsPromotableIntegerType for ABI specific needs. The
  // only difference is that this consideres bit-precise integer types as well.
  bool isPromotableIntegerTypeForABI(clang::QualType Ty) const;
};

} // namespace cir

#endif
