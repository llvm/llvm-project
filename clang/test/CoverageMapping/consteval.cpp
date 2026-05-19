// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++20 -triple %itanium_abi_triple -main-file-name consteval.cpp %s | FileCheck %s

// Consteval functions should not have coverage mappings, as they are evaluated
// entirely at compile time and produce no runtime code.
// See https://github.com/llvm/llvm-project/issues/164448.

// CHECK-NOT: _Z1gv:
consteval int g() { return 0; }

struct S {
  // CHECK-NOT: _ZN1S1sEv:
  static consteval int s() { return 1; }
};

// CHECK-LABEL: main:
// CHECK-NEXT: File 0, [[@LINE+1]]:12 -> [[@LINE+5]]:2 = #0
int main() {
  [[maybe_unused]] auto i = g();
  [[maybe_unused]] auto j = S::s();
  return 0;
}
