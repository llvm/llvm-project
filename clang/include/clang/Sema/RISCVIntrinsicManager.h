//===- RISCVIntrinsicManager.h - RISC-V Intrinsic Handler -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RISCVIntrinsicManager, which handles RISC-V vector
// intrinsic functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_RISCVINTRINSICMANAGER_H
#define LLVM_CLANG_SEMA_RISCVINTRINSICMANAGER_H

namespace clang {
class Sema;
class LookupResult;
class IdentifierInfo;
class Preprocessor;

namespace sema {
class RISCVIntrinsicManager {
public:
  virtual ~RISCVIntrinsicManager() = default;

  // Create RISC-V intrinsic and insert into symbol table and return true if
  // found, otherwise return false.
  virtual bool CreateIntrinsicIfFound(LookupResult &LR, IdentifierInfo *II,
                                      Preprocessor &PP) = 0;
};
} // end namespace sema
} // end namespace clang

#endif
