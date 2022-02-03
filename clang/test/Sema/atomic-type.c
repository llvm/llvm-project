// RUN: %clang_cc1 %s -verify -fsyntax-only

// Basic parsing/Sema tests for _Atomic
// No operations are actually supported on objects of this type yet.
// The qualifier syntax is not supported yet.
_Atomic(int) t1;
_Atomic(int) *t2 = &t1;
void testf(void*);
void f(void) {
  _Atomic(_Atomic(int)*) t3;
  _Atomic(_Atomic(int)*) *t4[2] = { &t3, 0 };
  testf(t4);
}
extern _Atomic(int (*)(int(*)[], int(*)[10])) mergetest;
extern _Atomic(int (*)(int(*)[10], int(*)[])) mergetest;
extern _Atomic(int (*)(int(*)[10], int(*)[10])) mergetest;

_Atomic(int(void)) error1; // expected-error {{_Atomic cannot be applied to function type}}
_Atomic(struct ErrorS) error2; // expected-error {{_Atomic cannot be applied to incomplete type}} expected-note {{forward declaration}}
_Atomic(int[10]) error3; // expected-error {{_Atomic cannot be applied to array type}}
_Atomic(const int) error4; // expected-error {{_Atomic cannot be applied to qualified type}}
_Atomic(_Atomic(int)) error5; // expected-error {{_Atomic cannot be applied to atomic type}}
