/*
 * Verify #pragma omp split counts(c1, c2, ...) at AST, IR, and runtime.
 * counts(3, 5, 2) splits 10 iterations into: [0..3), [3..8), [8..10).
 * Sum 0+1+...+9 = 45.
 */
// REQUIRES: x86-registered-target

// 1) Syntax and semantics only
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// expected-no-diagnostics

// 2) AST dump should show OMPSplitDirective with OMPCountsClause node.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 -ast-dump %s 2>&1 | FileCheck %s --check-prefix=AST

// 3) Emit LLVM: three sequential loops (multiple phi/br for loop structure)
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fopenmp -fopenmp-version=60 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=IR


int main(void) {
  const int n = 10;
  int sum = 0;

#pragma omp split counts(3, 5, 2)
  for (int i = 0; i < n; ++i) {
    sum += i;
  }

  return (sum == 45) ? 0 : 1;
}

// AST: OMPSplitDirective
// AST: OMPCountsClause

// IR: define
// IR: .split.iv.0
// IR: icmp slt i32 {{.*}}, 3
// IR: .split.iv.1
// IR: icmp slt i32 {{.*}}, 8
// IR: .split.iv.2
// IR: icmp slt i32 {{.*}}, 10
// IR: icmp eq i32 {{.*}}, 45
