// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage -fblocks -include %s -verify %s

// RUN: %clang -x c++ -frtti -fsyntax-only -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++11 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// RUN: %clang_cc1 -std=c++20 -fblocks -include %s %s 2>&1 | FileCheck --allow-empty %s
// CHECK-NOT: [-Wunsafe-buffer-usage]

#ifndef INCLUDED
#define INCLUDED
#pragma clang system_header

// no spanification warnings for system headers
void foo(...);  // let arguments of `foo` to hold testing expressions
#else

namespace std {
  class type_info;
  class bad_cast;
  class bad_typeid;
}
using size_t = __typeof(sizeof(int));
void *malloc(size_t);

void foo(int v) {
}

void foo(int *p){}

void uneval_context_fix() {
  auto p = new int[10]; // expected-warning{{'p' is an unsafe pointer used for buffer access}}

  // Warn on the following DREs
  _Generic(1, int: p[2], float: 3); // expected-note{{used in buffer access here}}

  // Do not warn for following DREs
  auto q = new int[10];
  foo(sizeof(q[1]), // no-note
      sizeof(decltype(q[1]))); // no-note
  __typeof(q[5]) x; // no-note
  int *r = (int *)malloc(sizeof(q[5])); // no-note
  int y = sizeof(q[5]); // no-note
  __is_pod(__typeof(q[5])); // no-note
  __is_trivially_constructible(__typeof(q[5]), decltype(q[5])); // no-note
  _Generic(q[1], int: 2, float: 3); // no-note
  _Generic(1, int: 2, float: q[3]); // no-note
  decltype(q[2]) var = y; // no-note
  noexcept(q[2]); // no-note
  typeid(q[3]); // no-note
}
#endif
