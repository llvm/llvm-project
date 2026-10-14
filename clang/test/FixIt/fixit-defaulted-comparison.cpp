// RUN: %clang_cc1 -verify -std=c++23 %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -std=c++23 -x c++ -fixit %t
// RUN: %clang_cc1 -std=c++23 -x c++ %t
// RUN: not %clang_cc1 -std=c++23 -x c++ -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

namespace std {
struct partial_ordering {};
} // namespace std

struct Box {
  std::partial_ordering operator<=>(const Box& other) = default; // #ssdecl
  bool operator==(const Box& other) = default; // #eqdecl
  // expected-error@#ssdecl {{defaulted member three-way comparison operator must be const-qualified}}
  // expected-error@#eqdecl {{defaulted member equality comparison operator must be const-qualified}}
  // CHECK: fix-it:{{.*}}:{12:54-12:54}:" const"
  // CHECK: fix-it:{{.*}}:{13:36-13:36}:" const"
};
