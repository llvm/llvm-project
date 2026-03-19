// RUN: %check_clang_tidy -check-header %S/Inputs/pass-by-value/header-with-fix.h \
// RUN:   %s modernize-pass-by-value %t -- -- -std=c++11

#include "header-with-fix.h"

// CHECK-MESSAGES: :[[@LINE+1]]:10: warning: pass by value and use std::move [modernize-pass-by-value]
Foo::Foo(const S &s) : s(s) {}
// CHECK-FIXES: #include <utility>
// CHECK-FIXES: Foo::Foo(S s) : s(std::move(s)) {}
