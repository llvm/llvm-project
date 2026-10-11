//===- DynamicAllocationArgumentsCXX.h - operator new/delete args ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the argument candidate and resolution types for operators
// new and new[] overload resolution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_DYNAMICALLOCATIONARGUMENTSCXX_H
#define LLVM_CLANG_SEMA_DYNAMICALLOCATIONARGUMENTSCXX_H

#include "clang/AST/ExprCXX.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class LookupResult;
class Sema;

struct ImplicitAllocationArguments {
  friend Sema;

  ArrayRef<Expr *> getImplicitArguments() const {
    return ArrayRef(ImplicitArguments, ArgumentCount);
  }

  Expr *getAlignmentArgument() const {
    if (PassAlignment == AlignedAllocationMode::Yes)
      return ImplicitArguments[ArgumentCount - 1];
    return nullptr;
  }

  const LookupResult &
  updateLookupForMSVCCompatibility(Sema &, const LookupResult &,
                                   std::optional<LookupResult> &) const;
  TypeAwareAllocationMode PassTypeIdentity;
  AlignedAllocationMode PassAlignment;
  bool IsMSVCCompatibilityFallback;

private:
  ImplicitAllocationArguments(Sema &SemaRef, Expr *TypeIdentityArg,
                              Expr *SizeArg, Expr *AlignArg,
                              bool IsMSVCCompatibilityFallback);

  // Type-identity, size, and alignment
  static constexpr unsigned MaxImplicitArguments = 3;
  unsigned ArgumentCount;
  Expr *ImplicitArguments[MaxImplicitArguments];
};

struct ResolvedAllocation {
  FunctionDecl *OperatorNew;
  FunctionDecl *OperatorDelete;
  ImplicitAllocationParameters IAP;
  // type-identity, size, alignment, nothrow or other single placement
  // parameter
  SmallVector<Expr *, 4> Arguments;
};

} // namespace clang

#endif // LLVM_CLANG_SEMA_DYNAMICALLOCATIONARGUMENTSCXX_H
