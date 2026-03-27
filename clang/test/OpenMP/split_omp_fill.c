/* Split + counts with omp_fill: syntax, AST dump, ast-print, IR. */
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump %s | FileCheck %s --check-prefix=DUMP
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM

void body(int);

// PRINT-LABEL: void foo(
// DUMP-LABEL:  FunctionDecl {{.*}} foo
void foo(int n) {
  // PRINT:     #pragma omp split counts(3, omp_fill)
  // DUMP: OMPSplitDirective
  // DUMP-NEXT: |-OMPCountsClause
  // DUMP-NEXT: | |-IntegerLiteral {{.*}} 'int' 3
  // DUMP-NEXT: | `-{{.*}}
  // DUMP-NEXT: {{.*}}`-ForStmt
#pragma omp split counts(3, omp_fill)
  // PRINT: for (int i = 0; i < n; ++i)
  for (int i = 0; i < n; ++i)
    body(i);
}

// LLVM-LABEL: define {{.*}}void @foo(
// LLVM: .split.iv.0.i
// LLVM: icmp slt i32 {{.*}}, 3
// LLVM: call void @body(
// LLVM: store i32 3, ptr %.split.iv.1.i
// LLVM: icmp slt i32 {{.*}}, %{{.*}}
// LLVM: call void @body(
