// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s

int test0(void *ptr) {
  return __builtin_is_modifiable_lvalue();	 // expected-error {{too few arguments to function call, expected 1, have 0}}
}

int test1(void *ptr) {
  return __builtin_is_modifiable_lvalue(ptr);	 // ok
}

int test2(void *ptr) {
  return __builtin_is_modifiable_lvalue(ptr, 5); // expected-error {{too many arguments to function call, expected 1, have 2}}
}

int test_trash(void *ptr) {
  return __builtin_is_modifiable_lvalue(trash);	 // expected-error {{use of undeclared identifier}}
}
