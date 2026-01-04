// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

void test(void) {
  __complex__ double C;
  double D;
  C / D;
}
