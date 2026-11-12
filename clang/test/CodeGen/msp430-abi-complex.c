// REQUIRES: msp430-registered-target
// RUN: %clang -target msp430 -Os -S -o- %s | FileCheck %s

volatile int N;
volatile int i16_1, i16_2;
volatile long i32_1, i32_2;
volatile long long i64_1, i64_2;
volatile float f1, f2;
volatile double d1, d2;

_Static_assert(sizeof(int) == 2, "Assumption failed");
_Static_assert(sizeof(long) == 4, "Assumption failed");
_Static_assert(sizeof(long long) == 8, "Assumption failed");

void complex_i16_arg_first(int _Complex x, int n) {
// CHECK-LABEL: @complex_i16_arg_first
  i16_1 = __real__ x;
// CHECK-DAG: mov r12, &i16_1
  i16_2 = __imag__ x;
// CHECK-DAG: mov r13, &i16_2
  N = n;
// CHECK-DAG: mov r14, &N
// CHECK:     ret
}

void complex_i16_arg_second(int n, int _Complex x) {
// CHECK-LABEL: @complex_i16_arg_second
  N = n;
// CHECK-DAG: mov r12, &N
  i16_1 = __real__ x;
// CHECK-DAG: mov r13, &i16_1
  i16_2 = __imag__ x;
// CHECK-DAG: mov r14, &i16_2
// CHECK:     ret
}

void complex_i32_arg_first(long _Complex x, int n) {
// CHECK-LABEL: @complex_i32_arg_first
  i32_1 = __real__ x;
// CHECK-DAG: mov r12, &i32_1
// CHECK-DAG: mov r13, &i32_1+2
  i32_2 = __imag__ x;
// CHECK-DAG: mov r14, &i32_2
// CHECK-DAG: mov r15, &i32_2+2
  N = n;
// CHECK-DAG: mov 2(r1), &N
// CHECK:     ret
}

void complex_i32_arg_second(int n, long _Complex x) {
// CHECK-LABEL: @complex_i32_arg_second
  N = n;
// CHECK-DAG: mov r12, &N
  i32_1 = __real__ x;
// CHECK-DAG: mov 2(r1), &i32_1
// CHECK-DAG: mov 4(r1), &i32_1+2
  i32_2 = __imag__ x;
// CHECK-DAG: mov 6(r1), &i32_2
// CHECK-DAG: mov 8(r1), &i32_2+2
// CHECK:     ret
}

void complex_i64_arg_first(long long _Complex x, int n) {
// CHECK-LABEL: @complex_i64_arg_first
  i64_1 = __real__ x;
// CHECK-DAG: mov 2(r1), &i64_1
// CHECK-DAG: mov 4(r1), &i64_1+2
// CHECK-DAG: mov 6(r1), &i64_1+4
// CHECK-DAG: mov 8(r1), &i64_1+6
  i64_2 = __imag__ x;
// CHECK-DAG: mov 10(r1), &i64_2
// CHECK-DAG: mov 12(r1), &i64_2+2
// CHECK-DAG: mov 14(r1), &i64_2+4
// CHECK-DAG: mov 16(r1), &i64_2+6
  N = n;
// CHECK-DAG: mov r12, &N
// CHECK:     ret
}

void complex_i64_arg_second(int n, long long _Complex x) {
// CHECK-LABEL: @complex_i64_arg_second
  N = n;
// CHECK-DAG: mov r12, &N
  i64_1 = __real__ x;
// CHECK-DAG: mov 2(r1), &i64_1
// CHECK-DAG: mov 4(r1), &i64_1+2
// CHECK-DAG: mov 6(r1), &i64_1+4
// CHECK-DAG: mov 8(r1), &i64_1+6
  i64_2 = __imag__ x;
// CHECK-DAG: mov 10(r1), &i64_2
// CHECK-DAG: mov 12(r1), &i64_2+2
// CHECK-DAG: mov 14(r1), &i64_2+4
// CHECK-DAG: mov 16(r1), &i64_2+6
// CHECK:     ret
}

void complex_float_arg_first(float _Complex x, int n) {
// CHECK-LABEL: @complex_float_arg_first
  f1 = __real__ x;
// CHECK-DAG: mov r12, &f1
// CHECK-DAG: mov r13, &f1+2
  f2 = __imag__ x;
// CHECK-DAG: mov r14, &f2
// CHECK-DAG: mov r15, &f2+2
  N = n;
// CHECK-DAG: mov 2(r1), &N
// CHECK:     ret
}

void complex_float_arg_second(int n, float _Complex x) {
// CHECK-LABEL: @complex_float_arg_second
  N = n;
// CHECK-DAG: mov r12, &N
  f1 = __real__ x;
// CHECK-DAG: mov 2(r1), &f1
// CHECK-DAG: mov 4(r1), &f1+2
  f2 = __imag__ x;
// CHECK-DAG: mov 6(r1), &f2
// CHECK-DAG: mov 8(r1), &f2+2
// CHECK:     ret
}

void complex_double_arg_first(double _Complex x, int n) {
// CHECK-LABEL: @complex_double_arg_first
  d1 = __real__ x;
// CHECK-DAG: mov 2(r1), &d1
// CHECK-DAG: mov 4(r1), &d1+2
// CHECK-DAG: mov 6(r1), &d1+4
// CHECK-DAG: mov 8(r1), &d1+6
  d2 = __imag__ x;
// CHECK-DAG: mov 10(r1), &d2
// CHECK-DAG: mov 12(r1), &d2+2
// CHECK-DAG: mov 14(r1), &d2+4
// CHECK-DAG: mov 16(r1), &d2+6
  N = n;
// CHECK-DAG: mov r12, &N
// CHECK:     ret
}

void complex_double_arg_second(int n, double _Complex x) {
// CHECK-LABEL: @complex_double_arg_second
  d1 = __real__ x;
// CHECK-DAG: mov 2(r1), &d1
// CHECK-DAG: mov 4(r1), &d1+2
// CHECK-DAG: mov 6(r1), &d1+4
// CHECK-DAG: mov 8(r1), &d1+6
  d2 = __imag__ x;
// CHECK-DAG: mov 10(r1), &d2
// CHECK-DAG: mov 12(r1), &d2+2
// CHECK-DAG: mov 14(r1), &d2+4
// CHECK-DAG: mov 16(r1), &d2+6
  N = n;
// CHECK-DAG: mov r12, &N
// CHECK:     ret
}

int _Complex complex_i16_res(void) {
// CHECK-LABEL: @complex_i16_res
  int _Complex res;
  __real__ res = 0x1122;
// CHECK-DAG: mov #4386, r12
  __imag__ res = 0x3344;
// CHECK-DAG: mov #13124, r13
  return res;
// CHECK:     ret
}

long _Complex complex_i32_res(void) {
// CHECK-LABEL: @complex_i32_res
  long _Complex res;
  __real__ res = 0x11223344;
// CHECK-DAG: mov #13124, r12
// CHECK-DAG: mov #4386,  r13
  __imag__ res = 0x55667788;
// CHECK-DAG: mov #30600, r14
// CHECK-DAG: mov #21862, r15
  return res;
// CHECK:     ret
}

long long _Complex complex_i64_res(void) {
// CHECK-LABEL: @complex_i64_res
  long long _Complex res;
  __real__ res = 0x1122334455667788;
// CHECK-DAG: mov #30600,  0(r12)
// CHECK-DAG: mov #21862,  2(r12)
// CHECK-DAG: mov #13124,  4(r12)
// CHECK-DAG: mov #4386,   6(r12)
  __imag__ res = 0x99aabbccddeeff00;
// CHECK-DAG: mov #-256,   8(r12)
// CHECK-DAG: mov #-8722,  10(r12)
// CHECK-DAG: mov #-17460, 12(r12)
// CHECK-DAG: mov #-26198, 14(r12)
  return res;
// CHECK:     ret
}

float _Complex complex_float_res(void) {
// CHECK-LABEL: @complex_float_res
  float _Complex res;
  __real__ res = 1;
// CHECK-DAG: clr r12
// CHECK-DAG: mov #16256, r13
  __imag__ res = -1;
// CHECK-DAG: clr r14
// CHECK-DAG: mov #-16512, r15
  return res;
// CHECK:     ret
}

double _Complex complex_double_res(void) {
// CHECK-LABEL: @complex_double_res
  double _Complex res;
  __real__ res = 1;
// CHECK-DAG: clr 0(r12)
// CHECK-DAG: clr 2(r12)
// CHECK-DAG: clr 4(r12)
// CHECK-DAG: mov #16368, 6(r12)
  __imag__ res = -1;
// CHECK-DAG: clr 8(r12)
// CHECK-DAG: clr 10(r12)
// CHECK-DAG: clr 12(r12)
// CHECK-DAG: mov #-16400, 14(r12)
  return res;
// CHECK:     ret
}
