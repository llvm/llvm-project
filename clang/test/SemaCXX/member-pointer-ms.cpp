// RUN: %clang_cc1 -cxx-abi microsoft -fsyntax-only -verify %s

// Test that we reject pointers to members of incomplete classes (for now)
struct A; //expected-note{{forward declaration of 'A'}}
int A::*pai1; //expected-error{{incomplete type 'A'}}

// Test that we don't allow reinterpret_casts from pointers of one size to
// pointers of a different size.
struct A {};
struct B {};
struct C: A, B {};

void (A::*paf)();
void (C::*pcf)() = reinterpret_cast<void (C::*)()>(paf); //expected-error{{cannot reinterpret_cast from member pointer type}}
