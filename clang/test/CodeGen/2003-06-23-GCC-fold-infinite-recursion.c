// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

double Test(double A, double B, double C, double D) {
  return -(A-B) - (C-D);
}

