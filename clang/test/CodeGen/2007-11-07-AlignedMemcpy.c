// RUN: %clang_cc1 -emit-llvm %s -o /dev/null
void bork(void) {
  int Qux[33] = {0};
}
