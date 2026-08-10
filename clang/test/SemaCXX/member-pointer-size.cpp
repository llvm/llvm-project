// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -fsyntax-only -verify
// RUN: %clang_cc1 -triple i686-unknown-unknown %s -fsyntax-only -verify
// expected-no-diagnostics
#include <stddef.h>

struct A;

void f() {
  int A::*dataMember;
  
  int (A::*memberFunction)();
  
  typedef int assert1[sizeof(dataMember) == sizeof(ptrdiff_t) ? 1 : -1];
  typedef int assert2[sizeof(memberFunction) == sizeof(ptrdiff_t) * 2 ? 1 : -1];
}

