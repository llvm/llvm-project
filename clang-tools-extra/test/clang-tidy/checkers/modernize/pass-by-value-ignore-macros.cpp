// RUN: %check_clang_tidy -check-suffixes=DEFAULT %s modernize-pass-by-value %t -- -config="{CheckOptions: {modernize-pass-by-value.IgnoreMacros: false}}"
// RUN: %check_clang_tidy -check-suffixes=IGNORE %s modernize-pass-by-value %t -- -config="{CheckOptions: {modernize-pass-by-value.IgnoreMacros: true}}"

struct A {
  A(const A &);
  A(A &&);
};

#define MACRO_CTOR(TYPE) \
struct TYPE { \
  TYPE(const A &a) : a(a) {} \
  A a; \
};
// CHECK-MESSAGES-DEFAULT: :[[@LINE+3]]:1: warning: pass by value and use std::move [modernize-pass-by-value]
// CHECK-MESSAGES-IGNORE-NOT: warning: pass by value and use std::move

MACRO_CTOR(B)

struct C {
  C(const A &a) : a(a) {}
  A a;
};
// CHECK-MESSAGES-DEFAULT: :[[@LINE-3]]:5: warning: pass by value and use std::move [modernize-pass-by-value]
// CHECK-MESSAGES-IGNORE: :[[@LINE-4]]:5: warning: pass by value and use std::move [modernize-pass-by-value]
