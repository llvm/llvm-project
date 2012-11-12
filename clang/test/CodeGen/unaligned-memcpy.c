// RUN: %clang_cc1 %s -emit-llvm -o -

void bork() {
  char Qux[33] = {0};
}
