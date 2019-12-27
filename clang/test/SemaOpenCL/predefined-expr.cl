// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 %s -verify -cl-std=CL2.0

void f() {
  char *f1 = __func__;          //expected-error-re{{initializing '{{__generic|__private}} char *__private' with an expression of type 'const __constant char *' changes address space of pointer}}
  constant char *f2 = __func__; //expected-warning{{initializing '__constant char *__private' with an expression of type 'const __constant char [2]' discards qualifiers}}
  constant const char *f3 = __func__;
}
