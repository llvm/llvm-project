// RUN: %clang_cc1 -triple %itanium_abi_triple-only %s -emit-llvm -o - | FileCheck %s

// Test that `annotate_type` does not affect mangled names.

int *[[clang::annotate_type("foo")]] f(int *[[clang::annotate_type("foo")]],
                                       int [[clang::annotate_type("foo")]]) {
  return nullptr;
}
// CHECK: @_Z1fPii

template <class T> struct S {};

S<int *[[clang::annotate_type("foo")]]>
g(S<int *[[clang::annotate_type("foo")]]>) {
  return {};
}
// CHECK: @_Z1g1SIPiE
