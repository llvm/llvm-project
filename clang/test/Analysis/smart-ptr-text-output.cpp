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
  std::unique_ptr<A> P(new A()); // expected-note {{Smart pointer 'P' is constructed}}
  // FIXME: should mark region as uninteresting after release, so above note will not be there
  P.release(); // expected-note {{Smart pointer 'P' is released and set to null}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterReset() {
  std::unique_ptr<A> P(new A()); // expected-note {{Smart pointer 'P' is constructed}}
  P.reset(); // expected-note {{Smart pointer 'P' reset using a null value}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefAfterResetWithNull() {
  A *NullInnerPtr = nullptr; // expected-note {{'NullInnerPtr' initialized to a null pointer value}}
  std::unique_ptr<A> P(new A()); // expected-note {{Smart pointer 'P' is constructed}}
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
  std::unique_ptr<A> P(new A()); // expected-note {{Smart pointer 'P' is constructed}}
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  P.swap(PNull); // expected-note {{Swapped null smart pointer 'PNull' with smart pointer 'P'}}
  PNull->foo(); // No warning.
  (*P).foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

// FIXME: Fix this test when "std::swap" is modeled seperately.
void derefOnStdSwappedNullPtr() {
  std::unique_ptr<A> P; // expected-note {{Default constructed smart pointer 'P' is null}}
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  std::swap(P, PNull); // expected-note@Inputs/system-header-simulator-cxx.h:979 {{Swapped null smart pointer 'PNull' with smart pointer 'P'}}
  // expected-note@-1 {{Calling 'swap<A>'}}
  // expected-note@-2 {{Returning from 'swap<A>'}}
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

void derefOnRawPtrFromGetOnNullPtr() {
  std::unique_ptr<A> P; // FIXME: add note "Default constructed smart pointer 'P' is null"
  P.get()->foo(); // expected-warning {{Called C++ object pointer is null [core.CallAndMessage]}}
  // expected-note@-1 {{Called C++ object pointer is null}}
}

void derefOnRawPtrFromGetOnValidPtr() {
  std::unique_ptr<A> P(new A());
  P.get()->foo(); // No warning.
}

void derefOnRawPtrFromGetOnUnknownPtr(std::unique_ptr<A> P) {
  P.get()->foo(); // No warning.
}

void derefOnMovedFromValidPtr() {
  std::unique_ptr<A> PToMove(new A());  // expected-note {{Smart pointer 'PToMove' is constructed}}
  // FIXME: above note should go away once we fix marking region not interested. 
  std::unique_ptr<A> P;
  P = std::move(PToMove); // expected-note {{Smart pointer 'PToMove' is null after being moved to 'P'}}
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1 {{Dereference of null smart pointer 'PToMove'}}
}

void derefOnMovedToNullPtr() {
  std::unique_ptr<A> PToMove(new A());
  std::unique_ptr<A> P;
  P = std::move(PToMove); // No note.
  P->foo(); // No warning.
}

void derefOnNullPtrGotMovedFromValidPtr() {
  std::unique_ptr<A> P(new A()); // expected-note {{Smart pointer 'P' is constructed}}
  // FIXME: above note should go away once we fix marking region not interested. 
  std::unique_ptr<A> PToMove; // expected-note {{Default constructed smart pointer 'PToMove' is null}}
  P = std::move(PToMove); // expected-note {{A null pointer value is moved to 'P'}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1 {{Dereference of null smart pointer 'P'}}
}

void derefOnMovedUnknownPtr(std::unique_ptr<A> PToMove) {
  std::unique_ptr<A> P;
  P = std::move(PToMove); // expected-note {{Smart pointer 'PToMove' is null after; previous value moved to 'P'}}
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1 {{Dereference of null smart pointer 'PToMove'}}
}

void derefOnAssignedNullPtrToNullSmartPtr() {
  std::unique_ptr<A> P; // expected-note {{Default constructed smart pointer 'P' is null}}
  P = nullptr; // expected-note {{Smart pointer 'P' is assigned to null}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1 {{Dereference of null smart pointer 'P'}}
}

void derefOnAssignedZeroToNullSmartPtr() {
  std::unique_ptr<A> P(new A()); // expected-note {{Smart pointer 'P' is constructed}}
  // FIXME: above note should go away once we fix marking region not interested. 
  P = 0; // expected-note {{Smart pointer 'P' is assigned to null}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1 {{Dereference of null smart pointer 'P'}}
}

void derefMoveConstructedWithNullPtr() {
  std::unique_ptr<A> PToMove; // expected-note {{Default constructed smart pointer 'PToMove' is null}}
  std::unique_ptr<A> P(std::move(PToMove)); // expected-note {{A null pointer value is moved to 'P'}}
  P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'P'}}
}

void derefValidPtrMovedToConstruct() {
  std::unique_ptr<A> PToMove(new A()); // expected-note {{Smart pointer 'PToMove' is constructed}}
  // FIXME: above note should go away once we fix marking region not interested. 
  std::unique_ptr<A> P(std::move(PToMove)); // expected-note {{Smart pointer 'PToMove' is null after being moved to 'P'}}
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'PToMove'}}
}

void derefNullPtrMovedToConstruct() {
  std::unique_ptr<A> PToMove; // expected-note {{Default constructed smart pointer 'PToMove' is null}}
  // FIXME: above note should go away once we fix marking region not interested. 
  std::unique_ptr<A> P(std::move(PToMove)); // expected-note {{Smart pointer 'PToMove' is null after being moved to 'P'}}
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'PToMove'}}
}

void derefUnknownPtrMovedToConstruct(std::unique_ptr<A> PToMove) {
  std::unique_ptr<A> P(std::move(PToMove)); // expected-note {{Smart pointer 'PToMove' is null after; previous value moved to 'P'}}
  PToMove->foo(); // expected-warning {{Dereference of null smart pointer 'PToMove' [alpha.cplusplus.SmartPtr]}}
  // expected-note@-1{{Dereference of null smart pointer 'PToMove'}}
}

void derefConditionOnNullPtrFalseBranch() {
  std::unique_ptr<A> P; // expected-note {{Default constructed smart pointer 'P' is null}}
  if (P) { // expected-note {{Taking false branch}}
    P->foo(); // No warning.
  } else {
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'P'}}
  }
}

void derefConditionOnNullPtrTrueBranch() {
  std::unique_ptr<A> P; // expected-note {{Default constructed smart pointer 'P' is null}}
  if (!P) { // expected-note {{Taking true branch}}
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'P'}}
  }
}

void derefConditionOnValidPtrTrueBranch() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  if (P) { // expected-note {{Taking true branch}}
    PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'PNull'}}
  } else {
    PNull->foo(); // No warning
  }
}

void derefConditionOnValidPtrFalseBranch() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  if (!P) { // expected-note {{Taking false branch}}
    PNull->foo(); // No warning
  } else {
    PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'PNull'}}
  }
}

void derefConditionOnNotValidPtr() {
  std::unique_ptr<A> P(new A());
  std::unique_ptr<A> PNull;
  if (!P)
    PNull->foo(); // No warning.
}

void derefConditionOnUnKnownPtrAssumeNull(std::unique_ptr<A> P) {
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  if (!P) { // expected-note {{Taking true branch}}
    // expected-note@-1{{Assuming smart pointer 'P' is null}}
    PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'PNull'}}
  }
}

void derefConditionOnUnKnownPtrAssumeNonNull(std::unique_ptr<A> P) {
  std::unique_ptr<A> PNull; // expected-note {{Default constructed smart pointer 'PNull' is null}}
  if (P) { // expected-note {{Taking true branch}}
    // expected-note@-1{{Assuming smart pointer 'P' is non-null}}
    PNull->foo(); // expected-warning {{Dereference of null smart pointer 'PNull' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'PNull'}}
  }
}

void derefOnValidPtrAfterReset(std::unique_ptr<A> P) {
  P.reset(new A());
  if (!P)
    P->foo(); // No warning.
  else
    P->foo(); // No warning.
}

struct S {
  std::unique_ptr<int> P;

  void foo() {
    if (!P) { // No-note because foo() is pruned
      return;
    }
  }

  int callingFooWithNullPointer() {
    foo(); // No note on Calling 'S::foo'
    P.reset(new int(0)); // expected-note {{Assigning 0}}
    return 1 / *(P.get()); // expected-warning {{Division by zero [core.DivideZero]}}
    // expected-note@-1 {{Division by zero}}
  }

  int callingFooWithValidPointer() {
    P.reset(new int(0)); // expected-note {{Assigning 0}}
    foo(); // No note on Calling 'S::foo'
    return 1 / *(P.get()); // expected-warning {{Division by zero [core.DivideZero]}}
    // expected-note@-1 {{Division by zero}}
  }

  int callingFooWithUnknownPointer(std::unique_ptr<int> PUnknown) {
    P.swap(PUnknown);
    foo(); // No note on Calling 'S::foo'
    P.reset(new int(0)); // expected-note {{Assigning 0}}
    return 1 / *(P.get()); // expected-warning {{Division by zero [core.DivideZero]}}
    // expected-note@-1 {{Division by zero}}
  }
};

void derefAfterBranchingOnUnknownInnerPtr(std::unique_ptr<A> P) {
  A *RP = P.get();
  if (!RP) { // expected-note {{Assuming 'RP' is null}}
    // expected-note@-1 {{Taking true branch}}
    P->foo(); // expected-warning {{Dereference of null smart pointer 'P' [alpha.cplusplus.SmartPtr]}}
    // expected-note@-1{{Dereference of null smart pointer 'P'}}
  }
}
