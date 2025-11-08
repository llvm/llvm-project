// RUN: not %clang_cc1 -std=c++20 -fsyntax-only %s 2>&1 | FileCheck %s

int main() {
  struct S {
    auto L = [](auto x) { return x; };
// CHECK: {{.+}}:[[@LINE-1]]:5: error: 'auto' not allowed in non-static struct member
    template<typename T> void memb(T);
// CHECK: {{.+}}:[[@LINE-1]]:5: error: templates cannot be declared inside of a local class
  };
  return 0;
}