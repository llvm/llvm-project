//===----- ABIInfo.h - ABI information access & encapsulation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_ABIINFO_H
#define LLVM_CLANG_LIB_CIR_ABIINFO_H

namespace clang::CIRGen {

class CIRGenFunctionInfo;
class CIRGenTypes;

class ABIInfo {
  ABIInfo() = delete;

public:
  CIRGenTypes &cgt;

  ABIInfo(CIRGenTypes &cgt) : cgt(cgt) {}

  virtual ~ABIInfo();

  virtual void computeInfo(CIRGenFunctionInfo &funcInfo) const = 0;
};

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_ABIINFO_H
