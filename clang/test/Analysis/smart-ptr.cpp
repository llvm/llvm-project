// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection\
// RUN:   -analyzer-checker cplusplus.Move,alpha.cplusplus.SmartPtr\
// RUN:   -analyzer-config cplusplus.SmartPtrModeling:ModelSmartPtrDereference=true\
// RUN:   -std=c++11 -verify %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_warnIfReached();
void clang_analyzer_numTimesReached();
void clang_analyzer_eval(bool);
void clang_analyzer_warnOnDeadSymbol(int *);

void derefAfterMove(std::unique_ptr<int> P) {
  std::unique_ptr<int> Q = std::move(P);
  if (Q)
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  *Q.get() = 1; // expected-warning {{Dereference of null pointer [core.NullDereference]}}
  if (P)
    clang_analyzer_warnIfReached(); // no-warning
  // TODO: Report a null dereference (instead).
  *P.get() = 1; // expected-warning {{Method called on moved-from object 'P' [cplusplus.Move]}}
  // expected-warning@-1 {{Dereference of null pointer [core.NullDereference]}}
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
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // No warning.
}

void derefOfUnknown(std::unique_ptr<A> P) {
  P->foo(); // No warning.
}

void derefAfterDefaultCtr() {
  std::unique_ptr<A> P;
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterCtrWithNull() {
  std::unique_ptr<A> P(nullptr);
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  *P; // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterCtrWithNullVariable() {
  A *InnerPtr = nullptr;
  std::unique_ptr<A> P(InnerPtr);
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterRelease() {
  std::unique_ptr<A> P(new A());
  P.release();
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterReset() {
  std::unique_ptr<A> P(new A());
  P.reset();
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterResetWithNull() {
  std::unique_ptr<A> P(new A());
  P.reset(nullptr);
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterResetWithNonNull() {
  std::unique_ptr<A> P;
  P.reset(new A());
  clang_analyzer_numTimesReached(); // expected-warning {{1}}
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

void derefOnReleasedValidRawPtr() {
  std::unique_ptr<A> P(new A());
  A *AP = P.release();
  AP->foo(); // No warning.
}

void pass_smart_ptr_by_ref(std::unique_ptr<A> &a);
void pass_smart_ptr_by_const_ref(const std::unique_ptr<A> &a);
void pass_smart_ptr_by_rvalue_ref(std::unique_ptr<A> &&a);
void pass_smart_ptr_by_const_rvalue_ref(const std::unique_ptr<A> &&a);
void pass_smart_ptr_by_ptr(std::unique_ptr<A> *a);
void pass_smart_ptr_by_const_ptr(const std::unique_ptr<A> *a);

void regioninvalidationWithPassByRef() {
  std::unique_ptr<A> P;
  pass_smart_ptr_by_ref(P);
  P->foo(); // no-warning
}

void regioninvalidationWithPassByCostRef() {
  std::unique_ptr<A> P;
  pass_smart_ptr_by_const_ref(P);
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void regioninvalidationWithPassByRValueRef() {
  std::unique_ptr<A> P;
  pass_smart_ptr_by_rvalue_ref(std::move(P));
  P->foo(); // no-warning
}

void regioninvalidationWithPassByConstRValueRef() {
  std::unique_ptr<A> P;
  pass_smart_ptr_by_const_rvalue_ref(std::move(P));
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void regioninvalidationWithPassByPtr() {
  std::unique_ptr<A> P;
  pass_smart_ptr_by_ptr(&P);
  P->foo();
}

void regioninvalidationWithPassByConstPtr() {
  std::unique_ptr<A> P;
  pass_smart_ptr_by_const_ptr(&P);
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

struct StructWithSmartPtr {
  std::unique_ptr<A> P;
};

void pass_struct_with_smart_ptr_by_ref(StructWithSmartPtr &a);
void pass_struct_with_smart_ptr_by_const_ref(const StructWithSmartPtr &a);
void pass_struct_with_smart_ptr_by_rvalue_ref(StructWithSmartPtr &&a);
void pass_struct_with_smart_ptr_by_const_rvalue_ref(const StructWithSmartPtr &&a);
void pass_struct_with_smart_ptr_by_ptr(StructWithSmartPtr *a);
void pass_struct_with_smart_ptr_by_const_ptr(const StructWithSmartPtr *a);

void regioninvalidationWithinStructPassByRef() {
  StructWithSmartPtr S;
  pass_struct_with_smart_ptr_by_ref(S);
  S.P->foo(); // no-warning
}

void regioninvalidationWithinStructPassByConstRef() {
  StructWithSmartPtr S;
  pass_struct_with_smart_ptr_by_const_ref(S);
  S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
}

void regioninvalidationWithinStructPassByRValueRef() {
  StructWithSmartPtr S;
  pass_struct_with_smart_ptr_by_rvalue_ref(std::move(S));
  S.P->foo(); // no-warning
}

void regioninvalidationWithinStructPassByConstRValueRef() {
  StructWithSmartPtr S;
  pass_struct_with_smart_ptr_by_const_rvalue_ref(std::move(S));
  S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
}

void regioninvalidationWithinStructPassByPtr() {
  StructWithSmartPtr S;
  pass_struct_with_smart_ptr_by_ptr(&S);
  S.P->foo(); // no-warning
}

void regioninvalidationWithinStructPassByConstPtr() {
  StructWithSmartPtr S;
  pass_struct_with_smart_ptr_by_const_ptr(&S);
  S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterAssignment() {
  {
    std::unique_ptr<A> P(new A());
    std::unique_ptr<A> Q;
    Q = std::move(P);
    Q->foo(); // no-warning
  }
  {
    std::unique_ptr<A> P;
    std::unique_ptr<A> Q;
    Q = std::move(P);
    Q->foo(); // expected-warning {{Dereference of null smart pointer 'Q' [alpha.cplusplus.SmartPtr]}}
  }
}

void derefOnSwappedNullPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull;
  P.swap(PNull);
  PNull->foo(); // No warning.
  (*P).foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnFirstStdSwappedNullPtr() {
  std::unique_ptr<A> P;
  std::unique_ptr<A> PNull;
  std::swap(P, PNull);
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnSecondStdSwappedNullPtr() {
  std::unique_ptr<A> P;
  std::unique_ptr<A> PNull;
  std::swap(P, PNull);
  PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
}

void derefOnSwappedValidPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PValid(new A());
  P.swap(PValid);
  (*P).foo(); // No warning.
  PValid->foo(); // No warning.
  std::swap(P, PValid);
  P->foo(); // No warning.
  PValid->foo(); // No warning.
}

void derefOnRawPtrFromGetOnNullPtr() {
  std::unique_ptr<A> P;
  P.get()->foo(); // expected-warning {{Called C++ object pointer is null [core.CallAndMessage]}}
}

void derefOnRawPtrFromGetOnValidPtr() {
  std::unique_ptr<A> P(new A());
  P.get()->foo(); // No warning.
}

void derefOnRawPtrFromGetOnUnknownPtr(std::unique_ptr<A> P) {
  P.get()->foo(); // No warning.
}

void derefOnRawPtrFromMultipleGetOnUnknownPtr(std::unique_ptr<A> P) {
  A *X = P.get();
  A *Y = P.get();
  clang_analyzer_eval(X == Y); // expected-warning{{TRUE}}
  if (!X) {
    Y->foo(); // expected-warning {{Called C++ object pointer is null [core.CallAndMessage]}}
  }
}

void derefOnMovedFromValidPtr() {
  std::unique_ptr<A> PToMove(new A());
  std::unique_ptr<A> P;
  P = std::move(PToMove);
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
}

void derefOnMovedToNullPtr() {
  std::unique_ptr<A> PToMove(new A());
  std::unique_ptr<A> P;
  P = std::move(PToMove); // No note.
  P->foo(); // No warning.
}

void derefOnNullPtrGotMovedFromValidPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PToMove;
  P = std::move(PToMove);
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnMovedFromUnknownPtr(std::unique_ptr<A> PToMove) {
  std::unique_ptr<A> P;
  P = std::move(PToMove);
  P->foo(); // No warning.
}

void derefOnMovedUnknownPtr(std::unique_ptr<A> PToMove) {
  std::unique_ptr<A> P;
  P = std::move(PToMove);
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
}

void derefOnAssignedNullPtrToNullSmartPtr() {
  std::unique_ptr<A> P;
  P = nullptr;
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnAssignedZeroToNullSmartPtr() {
  std::unique_ptr<A> P(new A());
  P = 0;
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnAssignedNullToUnknowSmartPtr(std::unique_ptr<A> P) {
  P = nullptr;
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

std::unique_ptr<A> &&returnRValRefOfUniquePtr();

void drefOnAssignedNullFromMethodPtrValidSmartPtr() {
  std::unique_ptr<A> P(new A());
  P = returnRValRefOfUniquePtr();
  P->foo(); // No warning.
}

void derefMoveConstructedWithValidPtr() {
  std::unique_ptr<A> PToMove(new A());
  std::unique_ptr<A> P(std::move(PToMove));
  P->foo(); // No warning.
}

void derefMoveConstructedWithNullPtr() {
  std::unique_ptr<A> PToMove;
  std::unique_ptr<A> P(std::move(PToMove));
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefMoveConstructedWithUnknownPtr(std::unique_ptr<A> PToMove) {
  std::unique_ptr<A> P(std::move(PToMove));
  P->foo(); // No warning.
}

void derefValidPtrMovedToConstruct() {
  std::unique_ptr<A> PToMove(new A());
  std::unique_ptr<A> P(std::move(PToMove));
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
}

void derefNullPtrMovedToConstruct() {
  std::unique_ptr<A> PToMove;
  std::unique_ptr<A> P(std::move(PToMove));
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
}

void derefUnknownPtrMovedToConstruct(std::unique_ptr<A> PToMove) {
  std::unique_ptr<A> P(std::move(PToMove));
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
}

std::unique_ptr<A> &&functionReturnsRValueRef();

void derefMoveConstructedWithRValueRefReturn() {
  std::unique_ptr<A> P(functionReturnsRValueRef());
  P->foo(); // No warning.
}

void derefConditionOnNullPtr() {
  std::unique_ptr<A> P;
  if (P)
    P->foo(); // No warning.
  else
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefConditionOnNotNullPtr() {
  std::unique_ptr<A> P;
  if (!P)
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefConditionOnValidPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull;
  if (P)
    PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
}

void derefConditionOnNotValidPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull;
  if (!P)
    PNull->foo(); // No warning.
}

void derefConditionOnUnKnownPtr(std::unique_ptr<A> P) {
  if (P)
    P->foo(); // No warning.
  else
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnValidPtrAfterReset(std::unique_ptr<A> P) {
  P.reset(new A());
  if (!P)
    P->foo(); // No warning.
  else
    P->foo(); // No warning.
}

void innerPointerSymbolLiveness() {
  std::unique_ptr<int> P(new int());
  clang_analyzer_warnOnDeadSymbol(P.get());
  int *RP = P.release();
} // expected-warning{{SYMBOL DEAD}}

void boolOpCreatedConjuredSymbolLiveness(std::unique_ptr<int> P) {
  if (P) {
    int *X = P.get();
    clang_analyzer_warnOnDeadSymbol(X);
  }
} // expected-warning{{SYMBOL DEAD}}

void getCreatedConjuredSymbolLiveness(std::unique_ptr<int> P) {
  int *X = P.get();
  clang_analyzer_warnOnDeadSymbol(X);
  int Y;
  if (!P) {
    Y = *P.get(); // expected-warning {{Dereference of null pointer [core.NullDereference]}}
    // expected-warning@-1 {{SYMBOL DEAD}}
  }
}

int derefConditionOnUnKnownPtr(int *q) {
  std::unique_ptr<int> P(q);
  if (P)
    return *P; // No warning.
  else
    return *P; // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterBranchingOnUnknownInnerPtr(std::unique_ptr<A> P) {
  A *RP = P.get();
  if (!RP) {
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  }
}
