// RUN: not %clang_cc1 -std=c++20 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK20
// RUN: not %clang_cc1 -std=c++17 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=CHECK17

int outer() {
  struct Outer {
    void f() {
      struct Inner {
        template<typename T> void m(T);
// CHECK20: {{.+}}:[[@LINE-1]]:9: error: templates cannot be declared inside of a local class
// CHECK20: {{.+}}:[[@LINE-3]]:7: error: templates cannot be declared inside of a local class
        void bad(auto x) {}
// CHECK17: {{.+}}:[[@LINE-1]]:18: error: 'auto' not allowed in function prototype
      };
    }
  };
  return 0;
}