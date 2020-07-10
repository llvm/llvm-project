// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection\
// RUN:   -analyzer-checker cplusplus.Move,alpha.cplusplus.SmartPtr\
// RUN:   -analyzer-config cplusplus.SmartPtrModeling:ModelSmartPtrDereference=true\
// RUN:   -std=c++11 -verify %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();
void clang_analyzer_numTimesReached();

void derefAfterMove(std::unique_ptr<int> P) {
  std::unique_ptr<int> Q = std::move(P);
  if (Q)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  *Q.get() = 1;                     // no-warning
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

class A {
public:
  A(){};
  void foo();
};

A *return_null() {
  return nullptr;
}

void derefAfterValidCtr() {
  std::unique_ptr<A> P(new A());
  P->foo(); // No warning.
}

void derefOfUnknown(std::unique_ptr<A> P) {
  P->foo(); // No warning.
}

void derefAfterDefaultCtr() {
  std::unique_ptr<A> P;
  P->foo(); // expected-warning {{Dereference of null smart pointer [alpha.cplusplus.SmartPtr]}}
}

void derefAfterCtrWithNull() {
  std::unique_ptr<A> P(nullptr);
  *P; // expected-warning {{Dereference of null smart pointer [alpha.cplusplus.SmartPtr]}}
}

void derefAfterCtrWithNullReturnMethod() {
  std::unique_ptr<A> P(return_null());
  P->foo(); // expected-warning {{Dereference of null smart pointer [alpha.cplusplus.SmartPtr]}}
}

void derefAfterRelease() {
  std::unique_ptr<A> P(new A());
  P.release();
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // expected-warning {{Dereference of null smart pointer [alpha.cplusplus.SmartPtr]}}
}

void derefAfterReset() {
  std::unique_ptr<A> P(new A());
  P.reset();
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // expected-warning {{Dereference of null smart pointer [alpha.cplusplus.SmartPtr]}}
}

void derefAfterResetWithNull() {
  std::unique_ptr<A> P(new A());
  P.reset(nullptr);
  P->foo(); // expected-warning {{Dereference of null smart pointer [alpha.cplusplus.SmartPtr]}}
}

void derefAfterResetWithNonNull() {
  std::unique_ptr<A> P;
  P.reset(new A());
  P->foo(); // No warning.
}

void derefAfterReleaseAndResetWithNonNull() {
  std::unique_ptr<A> P(new A());
  P.release();
  P.reset(new A());
  P->foo(); // No warning.
}

void derefOnReleasedNullRawPtr() {
  std::unique_ptr<A> P;
  A *AP = P.release();
  AP->foo(); // expected-warning {{Called C++ object pointer is null [core.CallAndMessage]}}
}
