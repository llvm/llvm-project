// RUN: %clang_analyze_cc1\
// RUN:  -analyzer-checker=core,cplusplus.Move,alpha.cplusplus.SmartPtr\
// RUN:  -analyzer-config cplusplus.SmartPtrModeling:ModelSmartPtrDereference=true\
// RUN:  -analyzer-output=text -std=c++11 %s -verify=expected

#include "Inputs/system-header-simulator-cxx.h"

class A {
public:
  A(){};
  void foo();
};

A *return_null() {
  return nullptr;
}

void derefAfterDefaultCtr() {
  std::unique_ptr<A> P; // expected-note {{Default constructed smart pointer 'P' is null}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterCtrWithNull() {
  A *NullInnerPtr = nullptr; // expected-note {{'NullInnerPtr' initialized to a null pointer value}}
  std::unique_ptr<A> P(NullInnerPtr); // expected-note {{Smart pointer 'P' is constructed using a null value}}
  *P; // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterCtrWithNullVariable() {
  A *NullInnerPtr = nullptr; // expected-note {{'NullInnerPtr' initialized to a null pointer value}}
  std::unique_ptr<A> P(NullInnerPtr); // expected-note {{Smart pointer 'P' is constructed using a null value}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterRelease() {
  std::unique_ptr<A> P(new A());
  P.release(); // expected-note {{Smart pointer 'P' is released and set to null}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterReset() {
  std::unique_ptr<A> P(new A());
  P.reset(); // expected-note {{Smart pointer 'P' reset using a null value}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterResetWithNull() {
  A *NullInnerPtr = nullptr; // expected-note {{'NullInnerPtr' initialized to a null pointer value}}
  std::unique_ptr<A> P(new A());
  P.reset(NullInnerPtr); // expected-note {{Smart pointer 'P' reset using a null value}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

// FIXME: Fix this test when support is added for tracking raw pointer
// and mark the smart pointer as interesting based on that and add tags.
void derefOnReleasedNullRawPtr() {
  std::unique_ptr<A> P; // FIXME: add note "Default constructed smart pointer 'P' is null"
  A *AP = P.release(); // expected-note {{'AP' initialized to a null pointer value}}
  AP->foo(); // expected-warning {{Called C++ object pointer is null [core.CallAndMessage]}}
  // expected-note@-1{{Called C++ object pointer is null}}
}

void derefOnSwappedNullPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  P.swap(PNull); // expected-note {{Swapped null smart pointer 'PNull' with smart pointer 'P'}}
  PNull->foo(); // No warning.
  (*P).foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

// FIXME: Fix this test when "std::swap" is modeled seperately.
void derefOnStdSwappedNullPtr() {
  std::unique_ptr<A> P;
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  std::swap(P, PNull); // expected-note@Inputs/system-header-simulator-cxx.h:978 {{Swapped null smart pointer 'PNull' with smart pointer 'P'}}
  // expected-note@-1 {{Calling 'swap<A>'}}
  // expected-note@-2 {{Returning from 'swap<A>'}}
  PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'PNull'}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

struct StructWithSmartPtr { // expected-note {{Default constructed smart pointer 'S.P' is null}}
  std::unique_ptr<A> P;
};

void derefAfterDefaultCtrInsideStruct() {
  StructWithSmartPtr S; // expected-note {{Calling implicit default constructor for 'StructWithSmartPtr'}}
  // expected-note@-1 {{Returning from default constructor for 'StructWithSmartPtr'}}
  S.P->foo(); // expected-warning {{Dereference of null smart pointer 'S.P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'S.P'}}
}

void noNoteTagsForNonInterestingRegion() {
  std::unique_ptr<A> P; // expected-note {{Default constructed smart pointer 'P' is null}}
  std::unique_ptr<A> P1; // No note.
  std::unique_ptr<A> P2; // No note.
  P1.release(); // No note.
  P1.reset(); // No note.
  P1.swap(P2); // No note.
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void noNoteTagsForNonMatchingBugType() {
  std::unique_ptr<A> P; // No note.
  std::unique_ptr<A> P1; // No note.
  P1 = std::move(P); // expected-note {{Smart pointer 'P' of type 'std::unique_ptr' is reset to null when moved from}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' of type 'std::unique_ptr' [cplusplus.Move]}}
  // expected-note@-1 {{Dereference of null smart pointer 'P' of type 'std::unique_ptr'}}
}
