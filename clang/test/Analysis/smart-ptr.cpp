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
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterCtrWithNull() {
  std::unique_ptr<A> P(nullptr);
  *P; // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefAfterCtrWithNullVariable() {
  A *InnerPtr = nullptr;
  std::unique_ptr<A> P(InnerPtr);
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

void regioninvalidationTest() {
  {
    std::unique_ptr<A> P;
    pass_smart_ptr_by_ref(P);
    P->foo(); // no-warning
  }
  {
    std::unique_ptr<A> P;
    pass_smart_ptr_by_const_ref(P);
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  }
  {
    std::unique_ptr<A> P;
    pass_smart_ptr_by_rvalue_ref(std::move(P));
    P->foo(); // no-warning
  }
  {
    std::unique_ptr<A> P;
    pass_smart_ptr_by_const_rvalue_ref(std::move(P));
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  }
  {
    std::unique_ptr<A> P;
    pass_smart_ptr_by_ptr(&P);
    P->foo();
  }
  {
    std::unique_ptr<A> P;
    pass_smart_ptr_by_const_ptr(&P);
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  }
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

void regioninvalidationTestWithinStruct() {
  {
    StructWithSmartPtr S;
    pass_struct_with_smart_ptr_by_ref(S);
    S.P->foo(); // no-warning
  }
  {
    StructWithSmartPtr S;
    pass_struct_with_smart_ptr_by_const_ref(S);
    S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
  }
  {
    StructWithSmartPtr S;
    pass_struct_with_smart_ptr_by_rvalue_ref(std::move(S));
    S.P->foo(); // no-warning
  }
  {
    StructWithSmartPtr S;
    pass_struct_with_smart_ptr_by_const_rvalue_ref(std::move(S));
    S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
  }
  {
    StructWithSmartPtr S;
    pass_struct_with_smart_ptr_by_ptr(&S);
    S.P->foo();
  }
  {
    StructWithSmartPtr S;
    pass_struct_with_smart_ptr_by_const_ptr(&S);
    S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
  }
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
    // TODO: Fix test with expecting warning after '=' operator overloading modeling.
    Q->foo(); // no-warning
  }
}

void derefOnSwappedNullPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull;
  P.swap(PNull);
  PNull->foo(); // No warning.
  (*P).foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
}

void derefOnStdSwappedNullPtr() {
  std::unique_ptr<A> P;
  std::unique_ptr<A> PNull;
  std::swap(P, PNull);
  PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
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
