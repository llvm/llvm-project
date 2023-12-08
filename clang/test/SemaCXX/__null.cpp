// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -Wno-null-conversion -fsyntax-only -verify
// RUN: %clang_cc1 -triple i686-unknown-unknown %s -Wno-null-conversion -fsyntax-only -verify

void f() {
  int* i = __null;
  i = __null;
  int i2 = __null;

  // Verify statically that __null is the right size
  int a[sizeof(typeof(__null)) == sizeof(void*)? 1 : -1];
  
  // Verify that null is evaluated as 0.
  int b[__null ? -1 : 1];
}

struct A {};

void g() {
  (void)(0 ? __null : A()); // expected-error {{non-pointer operand type 'A' incompatible with NULL}}
  (void)(0 ? A(): __null); // expected-error {{non-pointer operand type 'A' incompatible with NULL}}
}
