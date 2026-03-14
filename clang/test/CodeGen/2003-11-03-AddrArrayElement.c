// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

// This should be turned into a tasty getelementptr instruction, not a nasty
// series of casts and address arithmetic.

char Global[100];

char *test1(unsigned i) {
  // CHECK: getelementptr
  return &Global[i];
}
