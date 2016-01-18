// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

void foo(void*);

void bar()
{
  // declaring a function pointer is an error
  void (*fptr)(int); // expected-error{{pointers to functions are not allowed}}

  // taking the address of a function is an error
  foo((void*)foo); // expected-error{{taking address of function is not allowed}}
  foo(&foo); // expected-error{{taking address of function is not allowed}}

  // just calling a function is correct
  foo(0);
}
