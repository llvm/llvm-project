// RUN: %clang_cc1 -emit-llvm %s -o /dev/null
void bork() {
  int Qux[33] = {0};
}
