// RUN: %clang_cc1 -emit-llvm -o - %s

int f0(void *a, void *b) {
  return a - b;
}
