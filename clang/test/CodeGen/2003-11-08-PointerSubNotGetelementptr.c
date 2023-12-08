// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

char *test(char* C) {
  // CHECK: getelementptr
  return C-1;   // Should turn into a GEP
}

int *test2(int* I) {
  return I-1;
}
