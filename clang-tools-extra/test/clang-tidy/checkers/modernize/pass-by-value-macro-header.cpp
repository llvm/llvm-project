// RUN: %check_clang_tidy %s modernize-pass-by-value %t -- -- -isystem %clang_tidy_headers

// CHECK-FIXES: #include <utility>

#define HEADER <./a.h>
#include HEADER

struct A {
  A(const A &) {}
  A(A &&) {}
};

struct B {
  B(const A &a) : a(a) {}
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: pass by value and use std::move [modernize-pass-by-value]
// CHECK-FIXES: B(A a) : a(std::move(a)) {}
  A a;
};
