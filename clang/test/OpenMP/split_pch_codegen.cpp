// PCH round-trip for AST dump/print and host IR (split + counts).
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK1
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -include-pch %t -ast-dump-all %s | FileCheck %s --check-prefix=DUMP
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -include-pch %t -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -O0 -include-pch %t -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK2

#ifndef HEADER
#define HEADER

extern "C" void body(int);

// PRINT-LABEL: void foo(
// DUMP-LABEL:  FunctionDecl {{.*}} foo
void foo(int n) {
  // PRINT:     #pragma omp split counts(3, omp_fill)
  // DUMP:      OMPSplitDirective
  // DUMP-NEXT:   OMPCountsClause
  // DUMP: IntegerLiteral {{.*}} 3
#pragma omp split counts(3, omp_fill)
  // DUMP: ForStmt
  for (int i = 0; i < n; ++i)
    body(i);
}

// CHECK1-LABEL: define {{.*}} @_Z3foo
// CHECK1: .split.iv
// CHECK1: icmp
// CHECK1: call void @body

// CHECK2-LABEL: define {{.*}} @_Z3foo
// CHECK2: .split.iv
// CHECK2: icmp
// CHECK2: call void @body

#endif /* HEADER */
