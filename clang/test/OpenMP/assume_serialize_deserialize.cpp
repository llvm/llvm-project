// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -fopenmp -triple x86_64-unknown-linux-gnu %t/AssumeMod.cppm -emit-module-interface -o %t/AssumeMod.pcm
// RUN: %clang_cc1 -std=c++20 -fopenmp -triple x86_64-unknown-linux-gnu %t/UseAssumeMod.cpp -fmodule-file=AssumeMod=%t/AssumeMod.pcm -ast-dump-all | FileCheck %t/AssumeMod.cppm

// expected-no-diagnostics

//--- AssumeMod.cppm
module;
export module AssumeMod;
export int foo(int y) {
  int x = -1;
#pragma omp assume holds(y == 5)
// CHECK: OMPAssumeDirective 0x{{.*}} <line:5:1, col:33>
// CHECK-NEXT: OMPHoldsClause 0x{{.*}} <col:20, col:32>
  {
    x = y;
  }
  return x;
}
//--- UseAssumeMod.cpp
import AssumeMod;

extern "C" int printf(const char* fmt, ...);

int main() {
  printf ("foo(5)=%d\n", foo (5));
  return 0;
}
