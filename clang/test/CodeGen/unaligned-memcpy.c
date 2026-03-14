// RUN: %clang_cc1 %s -emit-llvm -o -

void bork(void) {
  char Qux[33] = {0};
}
