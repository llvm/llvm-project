// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 %s -emit-llvm -include-pch %t -o - | FileCheck %s

// The first run line creates a pch, and since at that point HEADER is not
// defined, the only thing contained in the pch is the pragma. The second line
// then includes that pch, so HEADER is defined and the actual code is compiled.
// The check then makes sure that the pragma is in effect in the file that
// includes the pch.

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
#pragma clang optimize off

#else

int a;

void f(void) {
  a = 12345;
}

// Check that the function is decorated with optnone

// CHECK-DAG: @f() [[ATTRF:#[0-9]+]]
// CHECK-DAG: attributes [[ATTRF]] = { {{.*}}noinline{{.*}}optnone{{.*}} }

#endif
