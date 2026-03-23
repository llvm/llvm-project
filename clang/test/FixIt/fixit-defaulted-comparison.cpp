// RUN: cp %s %t
// RUN: not %clang_cc1 -std=c++23 -x c++ -fixit %t
// RUN: %clang_cc1 -std=c++23 -x c++ %t
// RUN: not %clang_cc1 -std=c++23 -x c++ -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#include <compare>

struct Box {
  std::partial_ordering operator<=>(const Box& other) = default;
  // expected-error@-1 {{defaulted member three-way comparison operator must be const-qualified}}
  // CHECK: fix-it:{{.*}}:{7:54-7:54}:" const"
};
