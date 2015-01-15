// RUN: %clang %s -g -gcolumn-info -S -emit-llvm -o - | FileCheck %s
// Checks that clang emits column information when -gcolumn-info is passed.

int foo(int a, int b) { int c = a + b;


  return c;
}

// Without column information we wouldn't change locations for b.
// CHECK:  !MDLocation(line: 4, column: 20,
