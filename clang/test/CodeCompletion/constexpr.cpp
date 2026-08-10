// RUN: %clang_cc1 -fsyntax-only -std=c++11 -code-completion-at=%s:%(line+11):9 %s -o - | FileCheck %s

// PR14381: need constexpr function bodies always, even if code-completing.
template<int> struct S;
template<> struct S<1> {
  typedef int type;
};
constexpr int f() {
  return 1;
}

S<f()>::
// CHECK: COMPLETION: type : type
