//===----- ABIInfo.h - ABI information access & encapsulation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_ABIINFO_H
#define LLVM_CLANG_LIB_CIR_ABIINFO_H

#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

namespace clang::CIRGen {

class CIRGenFunctionInfo;
class CIRGenTypes;

class ABIInfo {
  ABIInfo() = delete;

public:
  CIRGenTypes &cgt;

  ABIInfo(CIRGenTypes &cgt) : cgt(cgt) {}

  virtual ~ABIInfo();

  /// Returns the optimal vector memory type for the given vector type. For
  /// example, on certain targets, a three-element vector may be widened to
  /// four elements to improve memory-access performance. The returned type
  /// must preserve the element type and must not have fewer elements.
  virtual cir::VectorType
  getOptimalVectorMemoryType(cir::VectorType type,
                             const clang::LangOptions &langOpts) const;
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_ABIINFO_H
