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

class BasicValueFactory;
namespace nonloc {
class ConcreteFloat;
} // namespace nonloc

/// A safe wrapper around APFloat objects allocated and owned by
/// \c BasicValueFactory. This just wraps a common llvm::APFloat.
class APFloatPtr {
  using APFloat = llvm::APFloat;

public:
  APFloatPtr() = delete;
  APFloatPtr(const APFloatPtr &) = default;
  APFloatPtr &operator=(const APFloatPtr &) & = default;
  ~APFloatPtr() = default;

  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const APFloat *get() const { return Ptr; }
  /*implicit*/ operator const APFloat &() const { return *get(); }

  const APFloat &operator*() const { return *Ptr; }
  const APFloat *operator->() const { return Ptr; }

private:
  /// \p Ptr is owned by \c BasicValueFactory, and \c nonloc::ConcreteFloat
  /// rewraps a pointer from that factory. Everyone else should use
  /// \c BasicValueFactory::getFloatValue() to get an APFloatPtr object.
  friend class BasicValueFactory;
  friend class nonloc::ConcreteFloat;

  explicit APFloatPtr(const APFloat *Ptr) : Ptr(Ptr) {}

  /// Owned by \c BasicValueFactory.
  const APFloat *Ptr;
};

} // namespace clang::ento

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_APFLOATPTR_H
