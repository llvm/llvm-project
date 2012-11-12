// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int test() {
  __complex__ double C;
  double D;
  C / D;
}
