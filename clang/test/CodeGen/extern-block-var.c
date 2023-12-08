// RUN: %clang_cc1 %s -emit-llvm -o %t

int f(void) {
  extern int a;
  return a;
}
