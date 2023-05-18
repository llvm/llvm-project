// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 %s -o - | FileCheck %s

int g();

// CHECK: _Z1fv(){{.*}} [[NR:#[0-9]+]]
[[noreturn]] int f() {
  while (g()) {}
}

// CHECK: attributes [[NR]] = { mustprogress noinline noreturn nounwind{{.*}} }
