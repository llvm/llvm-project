// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fsyntax-only -fixit -x c %t
// RUN: %clang_cc1 -fsyntax-only -pedantic -x c %t

void ip(int *aPtr) {}   // expected-note{{passing argument to parameter 'aPtr' here}}
void i(int a) {}        // expected-note{{passing argument to parameter 'a' here}}
void ii(int a) {}       // expected-note{{passing argument to parameter 'a' here}}
void fp(float *aPtr) {} // expected-note{{passing argument to parameter 'aPtr' here}}
void f(float a) {}      // expected-note{{passing argument to parameter 'a' here}}

void f2(int *aPtr, int a, float *bPtr, char c) {
  float fl = 0;
  ip(a);     // expected-warning{{incompatible integer to pointer conversion passing 'int' to parameter of type 'int *'; take the address with &}}
  i(aPtr);   // expected-warning{{incompatible pointer to integer conversion passing 'int *' to parameter of type 'int'; dereference with *}}
  ii(&a);     // expected-warning{{incompatible pointer to integer conversion passing 'int *' to parameter of type 'int'; remove &}}
  fp(*bPtr); // expected-error{{passing 'float' to parameter of incompatible type 'float *'; remove *}}
  f(bPtr);   // expected-error{{passing 'float *' to parameter of incompatible type 'float'; dereference with *}}
  a = aPtr;  // expected-warning{{incompatible pointer to integer conversion assigning to 'int' from 'int *'; dereference with *}}
  fl = bPtr + a;  // expected-error{{assigning to 'float' from incompatible type 'float *'; dereference with *}}
  bPtr = bPtr[a]; // expected-error{{assigning to 'float *' from incompatible type 'float'; take the address with &}}
}
