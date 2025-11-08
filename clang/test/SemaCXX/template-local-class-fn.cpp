// RUN: not %clang_cc1 -std=c++17 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK17
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK20
// RUN: not %clang_cc1 -std=c++23 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK23

int main() {
  struct Local {
// CHECK20: {{.+}}:[[@LINE-1]]:3: error: templates cannot be declared inside of a local class
// CHECK23: {{.+}}:[[@LINE-2]]:3: error: templates cannot be declared inside of a local class
    void mem(auto x) {}
// CHECK17: {{.+}}:[[@LINE-1]]:14: error: 'auto' not allowed in function prototype
  };
}