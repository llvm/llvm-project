// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -triple x86_64-apple-darwin13 -Wno-shift-count-overflow -verify=expected,cxx17 -std=c++17 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -triple x86_64-apple-darwin13 -Wno-shift-count-overflow -verify=expected,cxx2a -std=c++2a %s

int testNegativeShift(int a) {
  if (a == -5) {
    return 1 << a; // expected-warning{{The result of the left shift is undefined because the right operand is negative}}
  }
  return 0;
}

int testNegativeLeftShift(int a) {
  if (a == -3) {
    return a << 1; // cxx17-warning{{The result of the left shift is undefined because the left operand is negative}}
  }
  return 0;
}

int testUnrepresentableLeftShift(int a) {
  if (a == 8)
    return a << 30; // cxx17-warning{{The result of the left shift is undefined due to shifting '8' by '30', which is unrepresentable in the unsigned version of the return type 'int'}}
  return 0;
}
