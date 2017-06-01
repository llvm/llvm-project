// Test coverage_order_pcs=1 flag which orders the PCs by their appearance.
// RUN: DIR=%T/coverage-order-pcs
// RUN: rm -rf $DIR
// RUN: mkdir $DIR
// RUN: %clangxx_asan -fsanitize-coverage=func %s -o %t
// RUN: %env_asan_opts=coverage_dir=$DIR:coverage=1:coverage_order_pcs=0 %run %t
// RUN: mv $DIR/*sancov $DIR/A

// RUN: %env_asan_opts=coverage_dir=$DIR:coverage=1:coverage_order_pcs=0 %run %t 1
// RUN: mv $DIR/*sancov $DIR/B

// RUN: %env_asan_opts=coverage_dir=$DIR:coverage=1:coverage_order_pcs=1 %run %t
// RUN: mv $DIR/*sancov $DIR/C

// RUN: %env_asan_opts=coverage_dir=$DIR:coverage=1:coverage_order_pcs=1 %run %t 1
// RUN: mv $DIR/*sancov $DIR/D
//
// RUN: (%sancov print $DIR/A; %sancov print $DIR/B; %sancov print $DIR/C; %sancov print $DIR/D) | FileCheck %s
//
// RUN: rm -rf $DIR
// Ordering works only in 64-bit mode for now.
// REQUIRES: asan-64-bits, shell
// UNSUPPORTED: android
#include <stdio.h>

void foo() { fprintf(stderr, "FOO\n"); }
void bar() { fprintf(stderr, "BAR\n"); }

int main(int argc, char **argv) {
  if (argc == 2) {
    foo();
    bar();
  } else {
    bar();
    foo();
  }
}

// Run A: no ordering
// CHECK: [[FOO:0x[0-9a-f]*]]
// CHECK-NEXT: [[BAR:0x[0-9a-f]*]]
// CHECK-NEXT: [[MAIN:0x[0-9a-f]*]]
//
// Run B: still no ordering
// CHECK-NEXT: [[FOO]]
// CHECK-NEXT: [[BAR]]
// CHECK-NEXT: [[MAIN]]
//
// Run C: MAIN, BAR, FOO
// CHECK-NEXT: [[MAIN]]
// CHECK-NEXT: [[BAR]]
// CHECK-NEXT: [[FOO]]
//
// Run D: MAIN, FOO, BAR
// CHECK-NEXT: [[MAIN]]
// CHECK-NEXT: [[FOO]]
// CHECK-NEXT: [[BAR]]
