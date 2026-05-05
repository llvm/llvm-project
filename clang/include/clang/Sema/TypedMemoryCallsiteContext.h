//===- TypedMemoryCallsiteContext.h - Context info for TMO calls -*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines TypedMemoryCallsiteContext, which contains semantic
// information about the TMO calls in the current scope.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_TYPEDMEMORYCALLSITECONTEXT_H
#define LLVM_CLANG_SEMA_TYPEDMEMORYCALLSITECONTEXT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class Expr;
class CallExpr;
class CastExpr;
class Sema;

namespace sema {

class TypedMemoryCallsiteContext {
  llvm::SmallVector<const CallExpr *, 4> Calls;
  llvm::DenseMap<const CallExpr *, const CastExpr *> Casts;
  bool ShouldSearchCasts = false;

  void recordInfoForInferredCall(Sema &S, const CallExpr *Call);

public:
  void clear() {
    ShouldSearchCasts = false;
    Calls.clear();
    Casts.clear();
  }

  void finalizeOutstandingTMOCandidates(Sema &);
  void recordTMOInferenceCandidate(Sema &, const Expr *);
  void recordCastForTMOInference(Sema &, const CastExpr *Cast);
};

} // namespace sema

} // namespace clang

#endif // LLVM_CLANG_SEMA_TYPEDMEMORYCALLSITECONTEXT_H
