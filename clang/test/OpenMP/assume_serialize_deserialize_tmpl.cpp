// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -fopenmp -triple x86_64-unknown-linux-gnu %t/AssumeMod.cppm -emit-module-interface -o %t/AssumeMod.pcm
// RUN: %clang_cc1 -std=c++20 -fopenmp -triple x86_64-unknown-linux-gnu %t/UseAssumeMod.cpp -fmodule-file=AssumeMod=%t/AssumeMod.pcm -ast-dump-all | FileCheck %t/AssumeMod.cppm

// expected-no-diagnostics

//--- AssumeMod.cppm
module;
export module AssumeMod;
export template<int N> int foo(int y) {
  int x = -1;
#pragma omp assume holds(y == N)
  {
    x = y;
  }
  return x;
}

export template<bool B> void bar(int *z) {
  if constexpr (B) {
#pragma omp assume no_openmp
    {
      z[0]++;
    }
  } else {
#pragma omp assume contains(target, parallel) absent(loop)
    {
      z[1] += 2;
    }
  }
}

// CHECK: FunctionTemplateDecl 0x{{.*}}
// CHECK: OMPAssumeDirective 0x{{.*}} <line:5:1, col:33>
// CHECK-NEXT: OMPHoldsClause 0x{{.*}} <col:20, col:32>

// CHECK: FunctionDecl 0x{{.*}} implicit_instantiation
// CHECK: OMPAssumeDirective 0x{{.*}} <line:5:1, col:33>
// CHECK-NEXT: OMPHoldsClause 0x{{.*}} <col:20, col:32>

// CHECK: FunctionTemplateDecl 0x{{.*}}
// CHECK: OMPAssumeDirective 0x{{.*}} <line:14:1, col:29>
// CHECK-NEXT: OMPNo_openmpClause 0x{{.*}} <col:20, col:29>
// CHECK: OMPAssumeDirective 0x{{.*}} <line:19:1, col:59>
// CHECK-NEXT: OMPContainsClause 0x{{.*}} <col:20, col:45>
// CHECK-NEXT: OMPAbsentClause 0x{{.*}} <col:47, col:58>

// CHECK: FunctionDecl 0x{{.*}} implicit_instantiation
// CHECK: OMPAssumeDirective 0x{{.*}} <line:14:1, col:29>
// CHECK-NEXT: OMPNo_openmpClause 0x{{.*}} <col:20, col:29>

// CHECK: FunctionDecl 0x{{.*}} implicit_instantiation
// CHECK: OMPAssumeDirective 0x{{.*}} <line:19:1, col:59>
// CHECK-NEXT: OMPContainsClause 0x{{.*}} <col:20, col:45>
// CHECK-NEXT: OMPAbsentClause 0x{{.*}} <col:47, col:58>

//--- UseAssumeMod.cpp
import AssumeMod;

extern "C" int printf(const char* fmt, ...);

int main() {
  printf ("foo(5)=%d\n", foo<5> (5));
  int arr[2] = { 0, 0 };
  bar<true>(arr);
  bar<false>(arr);
  printf ("arr[0]=%d, arr[1]=%d\n", arr[0], arr[1]);
  return 0;
}
