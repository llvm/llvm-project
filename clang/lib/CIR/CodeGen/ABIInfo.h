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
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace clang::CIRGen {

class CIRGenFunctionInfo;
class CIRGenTypes;

class ABIInfo {
  ABIInfo() = delete;

public:
  CIRGenTypes &cgt;

  ABIInfo(CIRGenTypes &cgt) : cgt(cgt) {}

  virtual ~ABIInfo();

  /// Returns the optimal vector memory type based on the given vector type. For
  /// example, on certain targets, a vector with 3 elements might be promoted to
  /// one with 4 elements to improve performance.
  virtual cir::VectorType
  getOptimalVectorMemoryType(cir::VectorType ty, const LangOptions &opt) const;
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_ABIINFO_H
