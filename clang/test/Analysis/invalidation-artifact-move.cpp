// RUN: %clang_analyze_cc1 -std=c++17 -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection,cplusplus.Move,alpha.cplusplus.IteratorModeling \
// RUN:   -analyzer-config aggressive-binary-operation-simplification=true \
// RUN:   -analyzer-config c++-container-inlining=true

// MoveChecker::evalCall models the 3-argument std::move algorithm. As part of
// that partial modeling it hands the destination container off to
// ProgramState::invalidateRegions. After this patch the resulting symbols
// carry a "partial-call" cause instead of a plain conservative-eval conj_$.

#include "Inputs/system-header-simulator-cxx.h"

template <typename T> void clang_analyzer_dump(T);

void test_move_dest_invalidation() {
  std::vector<int> src;
  src.push_back(1);
  std::vector<int> dst;

  std::move(src.begin(), src.end(), std::back_inserter(dst));

  // The destination container's contents are now bound to an inv_$ artifact
  // carrying a "partial-call" cause.
  clang_analyzer_dump(dst[0]); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, partial-call, S[0-9]+, #[0-9]+}}}}}
}
