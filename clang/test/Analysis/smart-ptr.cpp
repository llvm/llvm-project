// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection\
// RUN:   -analyzer-checker cplusplus.Move,cplusplus.SmartPtr\
// RUN:   -std=c++11 -verify %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();

void derefAfterMove(std::unique_ptr<int> P) {
  std::unique_ptr<int> Q = std::move(P);
  if (Q)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  *Q.get() = 1; // no-warning
  if (P)
    clang_analyzer_warnIfReached(); // no-warning
  // TODO: Report a null dereference (instead).
  *P.get() = 1; // expected-warning {{Method called on moved-from object 'P'}}
}

// Don't crash when attempting to model a call with unknown callee.
namespace testUnknownCallee {
struct S {
  void foo();
};
void bar(S *s, void (S::*func)(void)) {
  (s->*func)(); // no-crash
}
} // namespace testUnknownCallee
