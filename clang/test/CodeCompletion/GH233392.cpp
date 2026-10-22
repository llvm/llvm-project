// Check that code completion inside OpenMP variable-list clauses terminates.
// https://github.com/llvm/llvm-project/issues/233392

// RUN: split-file %s %t
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/reduction.cpp:3:39 %t/reduction.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/reduction.cpp:3:40 %t/reduction.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/reduction.cpp:3:41 %t/reduction.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/logical_reduction.cpp:3:41 %t/logical_reduction.cpp | FileCheck -check-prefix=CHECK-X %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/empty_reduction.cpp:3:41 %t/empty_reduction.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/private.cpp:3:34 %t/private.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/private_variable.cpp:3:34 %t/private_variable.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/shared.cpp:3:33 %t/shared.cpp | FileCheck -check-prefix=CHECK-X %s
// RUN: %clang_cc1 -fsyntax-only -x c -fopenmp -code-completion-at=%t/reduction.c:3:40 %t/reduction.c | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -code-completion-at=%t/simd.cpp:3:32 %t/simd.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -code-completion-at=%t/reduction.cpp:3:40 %t/reduction.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/if.cpp:3:29 %t/if.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/num_threads.cpp:3:38 %t/num_threads.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/schedule.cpp:3:35 %t/schedule.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/reduction.cpp:4:32 %t/reduction.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%t/no_openmp.cpp:3:40 %t/no_openmp.cpp
// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=52 -code-completion-at=%t/linear.cpp:3:42 %t/linear.cpp | FileCheck -check-prefix=CHECK-S %s
// RUN: not %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/threadprivate.cpp:2:27 %t/threadprivate.cpp | FileCheck -check-prefix=CHECK-THREADPRIVATE %s
// RUN: %clang_cc1 -fsyntax-only -fopenmp -code-completion-at=%t/declare_reduction.cpp:1:35 %t/declare_reduction.cpp | FileCheck -check-prefix=CHECK-DECLRED %s

//--- reduction.cpp
void f() {
  int s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < 10; ++i) s += i;
}

//--- logical_reduction.cpp
void f() {
  int x = 0;
#pragma omp parallel for reduction(&& : x)
  for (int i = 0; i < 10; ++i) x += i;
}

//--- empty_reduction.cpp
void f() {
  int s = 0;
#pragma omp parallel for reduction(&& : )
  for (int i = 0; i < 10; ++i) s += i;
}

//--- private.cpp
void f() {
  int s = 0;
#pragma omp parallel for private()
  for (int i = 0; i < 10; ++i) s += i;
}

//--- private_variable.cpp
void f() {
  int s = 0;
#pragma omp parallel for private(s)
  for (int i = 0; i < 10; ++i) s += i;
}

//--- shared.cpp
void f() {
  int x = 0;
#pragma omp parallel for shared(x)
  for (int i = 0; i < 10; ++i) x += i;
}

//--- reduction.c
void f() {
  int s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < 10; ++i) s += i;
}

//--- simd.cpp
void f() {
  int s = 0;
#pragma omp simd reduction(+ : s)
  for (int i = 0; i < 10; ++i) s += i;
}

//--- if.cpp
void f() {
  int s = 0;
#pragma omp parallel for if()
  for (int i = 0; i < 10; ++i) s += i;
}

//--- num_threads.cpp
void f() {
  int s = 0;
#pragma omp parallel for num_threads()
  for (int i = 0; i < 10; ++i) s += i;
}

//--- schedule.cpp
void f() {
  int s = 0;
#pragma omp parallel for schedule()
  for (int i = 0; i < 10; ++i) s += i;
}

//--- no_openmp.cpp
void f() {
  int s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < 10; ++i) s += i;
}

//--- linear.cpp
void f() {
  int s = 0;
#pragma omp parallel for linear(s : step(2))
  for (int i = 0; i < 10; ++i) s += i;
}

//--- threadprivate.cpp
int tp;
#pragma omp threadprivate(tp)

//--- declare_reduction.cpp
#pragma omp declare reduction(+ : int : omp_out = omp_in)
void bar();

// CHECK-S: COMPLETION: s : [#int#]s
// CHECK-X: COMPLETION: x : [#int#]x
// CHECK-THREADPRIVATE: COMPLETION: Pattern : alignas(<#expression#>)
// CHECK-DECLRED: COMPLETION: int
