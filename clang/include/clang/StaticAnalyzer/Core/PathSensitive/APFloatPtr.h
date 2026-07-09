//== APFloatPtr.h - Wrapper for APFloat objects owned separately -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_APFLOATPTR_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_APFLOATPTR_H

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Compiler.h"

namespace clang::ento {

/// A safe wrapper around APFloat objects allocated and owned by
/// \c BasicValueFactory. This just wraps a common llvm::APFloat.
class APFloatPtr {
  using APFloat = llvm::APFloat;

public:
  APFloatPtr() = delete;
  APFloatPtr(const APFloatPtr &) = default;
  APFloatPtr &operator=(const APFloatPtr &) & = default;
  ~APFloatPtr() = default;

  /// You should not use this API.
  /// If do, ensure that the \p Ptr is not going to dangle.
  /// Prefer using \c BasicValueFactory::getFloatValue() to get an APFloatPtr
  /// object.
  static APFloatPtr unsafeConstructor(const APFloat *Ptr) {
    return APFloatPtr(Ptr);
  }

  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const APFloat *get() const { return Ptr; }
  /*implicit*/ operator const APFloat &() const { return *get(); }

  const APFloat &operator*() const { return *Ptr; }
  const APFloat *operator->() const { return Ptr; }

private:
  explicit APFloatPtr(const APFloat *Ptr) : Ptr(Ptr) {}

  /// Owned by \c BasicValueFactory.
  const APFloat *Ptr;
};

} // namespace clang::ento

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_APFLOATPTR_H
