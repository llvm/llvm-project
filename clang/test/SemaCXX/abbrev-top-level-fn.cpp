// RUN: not %clang_cc1 -std=c++17 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK17
// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s

void top_fun(auto x) { }
// CHECK17: {{.+}}:[[@LINE-1]]:14: error: 'auto' not allowed in function prototype