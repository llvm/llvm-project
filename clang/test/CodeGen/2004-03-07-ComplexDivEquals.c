// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


void test(__complex__ double D, double X) {
  D /= X;
}
