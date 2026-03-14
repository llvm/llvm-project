// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -x c++-header %s -emit-pch -o %t/a.pch
// RUN: %clang_cc1 -x c++-header %s -emit-pch -o %t/b.pch
// RUN: cmp %t/a.pch %t/b.pch

#pragma float_control(push)
double fp_control_0(double x) {
  return -x + x;
}

double fp_control_1(double x) {
#pragma float_control(precise, on)
  return -x + x;
}

double fp_control_2(double x) {
#pragma float_control(precise, off)
  return -x + x;
}
#pragma float_control(pop)
