// RUN: %clang_cc1 -emit-pch -o %t1.pch %s
// RUN: %clang_cc1 -emit-pch -o %t2.pch %s
// RUN: %clang_cc1 %s -include-pch %t1.pch -include-pch %t2.pch -verify

#ifndef HEADER
#define HEADER

extern int x;

#else

#warning parsed this
// expected-warning@-1 {{parsed this}}
int foo(void) {
  return x;
}

#endif
