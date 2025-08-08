//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes support the generation of CIR for cleanups, initially based
// on LLVM IR cleanup handling, but ought to change as CIR evolves.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H

#include "EHScopeStack.h"

namespace clang::CIRGen {

/// A non-stable pointer into the scope stack.
class EHScopeStack::iterator {
  char *ptr = nullptr;

  friend class EHScopeStack;
  explicit iterator(char *ptr) : ptr(ptr) {}

public:
  iterator() = default;

  EHScopeStack::Cleanup *get() const {
    return reinterpret_cast<EHScopeStack::Cleanup *>(ptr);
  }

  EHScopeStack::Cleanup &operator*() const { return *get(); }
};

inline EHScopeStack::iterator EHScopeStack::begin() const {
  return iterator(startOfData);
}

} // namespace clang::CIRGen
#endif // CLANG_LIB_CIR_CODEGEN_CIRGENCLEANUP_H
