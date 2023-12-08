// REQUIRES: arm-registered-target

// RUN: %clang_cc1 -triple armv6 -verify=arm6 %s
// RUN: %clang_cc1 -triple armv7 -verify=arm7 %s
// RUN: %clang_cc1 -triple thumbv6 -verify=thumb1 %s
// RUN: %clang_cc1 -triple thumbv7 -verify=thumb2 %s

// j: An immediate integer between 0 and 65535 (valid for MOVW) (ARM/Thumb2)
int test_j(int i) {
  int res;
  __asm("movw %0, %1;"
        : [ result ] "=r"(res)
        : [ constant ] "j"(-1), [ input ] "r"(i)
        :);
  // arm6-error@13 {{invalid input constraint 'j' in asm}}
  // arm7-error@13 {{value '-1' out of range for constraint 'j'}}
  // thumb1-error@13 {{invalid input constraint 'j' in asm}}
  // thumb2-error@13 {{value '-1' out of range for constraint 'j'}}
  __asm("movw %0, %1;"
        : [ result ] "=r"(res)
        : [ constant ] "j"(0), [ input ] "r"(i)
        :);
  // arm6-error@21 {{invalid input constraint 'j' in asm}}
  // arm7-no-error
  // thumb1-error@21 {{invalid input constraint 'j' in asm}}
  // thumb2-no-error
  __asm("movw %0, %1;"
        : [ result ] "=r"(res)
        : [ constant ] "j"(65535), [ input ] "r"(i)
        :);
  // arm6-error@29 {{invalid input constraint 'j' in asm}}
  // arm7-no-error
  // thumb1-error@29 {{invalid input constraint 'j' in asm}}
  // thumb2-no-error
  __asm("movw %0, %1;"
        : [ result ] "=r"(res)
        : [ constant ] "j"(65536), [ input ] "r"(i)
        :);
  // arm6-error@37 {{invalid input constraint 'j' in asm}}
  // arm7-error@37 {{value '65536' out of range for constraint 'j'}}
  // thumb1-error@37 {{invalid input constraint 'j' in asm}}
  // thumb2-error@37 {{value '65536' out of range for constraint 'j'}}
  return res;
}

// I: An immediate integer valid for a data-processing instruction. (ARM/Thumb2)
//    An immediate integer between 0 and 255. (Thumb1)
int test_I(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "I"(-1), [ input ] "r"(i)
      :); // thumb1-error@53 {{value '-1' out of range for constraint 'I'}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "I"(0), [ input ] "r"(i)
      :); // No errors expected.
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "I"(255), [ input ] "r"(i)
      :); // No errors expected.
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "I"(256), [ input ] "r"(i)
      :); // thumb1-error@68 {{value '256' out of range for constraint 'I'}}
  return res;
}

// J: An immediate integer between -4095 and 4095. (ARM/Thumb2)
//    An immediate integer between -255 and -1. (Thumb1)
int test_J(int i) {
  int res;
  __asm(
      "movw %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(-4096), [ input ] "r"(i)
      :);
  // arm6-error@80 {{value '-4096' out of range for constraint 'J'}}
  // arm7-error@80 {{value '-4096' out of range for constraint 'J'}}
  // thumb1-error@80 {{value '-4096' out of range for constraint 'J'}}
  // thumb2-error@80 {{value '-4096' out of range for constraint 'J'}}
  __asm(
      "movw %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(-4095), [ input ] "r"(i)
      :);
  // thumb1-error@89 {{value '-4095' out of range for constraint 'J'}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(-256), [ input ] "r"(i)
      :);
  // thumb1-error@95 {{value '-256' out of range for constraint 'J'}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(-255), [ input ] "r"(i)
      :);
  // No errors expected.
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(-1), [ input ] "r"(i)
      :);
  // No errors expected.
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(0), [ input ] "r"(i)
      :);
  // thumb1-error@113 {{value '0' out of range for constraint 'J'}}
  __asm(
      "movw %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(4095), [ input ] "r"(i)
      :);
  // thumb1-error@119 {{value '4095' out of range for constraint 'J'}}
  __asm(
      "movw %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "J"(4096), [ input ] "r"(i)
      :);
  // arm6-error@125 {{value '4096' out of range for constraint 'J'}}
  // arm7-error@125 {{value '4096' out of range for constraint 'J'}}
  // thumb1-error@125 {{value '4096' out of range for constraint 'J'}}
  // thumb2-error@125 {{value '4096' out of range for constraint 'J'}}
  return res;
}

// K: An immediate integer whose bitwise inverse is valid for a data-processing instruction. (ARM/Thumb2)
//    An immediate integer between 0 and 255, with optional left-shift by some amount. (Thumb1)
int test_K(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "K"(123), [ input ] "r"(i)
      :);
  // No errors expected.
  return res;
}

// L: An immediate integer whose negation is valid for a data-processing instruction. (ARM/Thumb2)
//    An immediate integer between -7 and 7. (Thumb1)
int test_L(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "L"(-8), [ input ] "r"(i)
      :); // thumb1-error@154 {{value '-8' out of range for constraint 'L'}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "L"(-7), [ input ] "r"(i)
      :); // No errors expected.
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "L"(7), [ input ] "r"(i)
      :); // No errors expected.
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "L"(8), [ input ] "r"(i)
      :); // thumb1-error@169 {{value '8' out of range for constraint 'L'}}
  return res;
}

// M: A power of two or a integer between 0 and 32. (ARM/Thumb2)
//    An immediate integer which is a multiple of 4 between 0 and 1020. (Thumb1)
int test_M(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "M"(123), [ input ] "r"(i)
      :); // No errors expected.
  return res;
}

// N: Invalid (ARM/Thumb2)
//    An immediate integer between 0 and 31. (Thumb1)
int test_N(int i) {
  int res;
  __asm("add %0, %1;"
        : [ result ] "=r"(res)
        : [ constant ] "N"(-1), [ input ] "r"(i)
        :);
  // arm6-error@192 {{invalid input constraint 'N' in asm}}
  // arm7-error@192 {{invalid input constraint 'N' in asm}}
  // thumb1-error@192 {{value '-1' out of range for constraint 'N'}}
  // thumb2-error@192 {{invalid input constraint 'N' in asm}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "N"(0), [ input ] "r"(i)
      :);
  // arm6-error@201 {{invalid input constraint 'N' in asm}}
  // arm7-error@201 {{invalid input constraint 'N' in asm}}
  // thumb1-no-error
  // thumb2-error@201 {{invalid input constraint 'N' in asm}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "N"(31), [ input ] "r"(i)
      :);
  // arm6-error@210 {{invalid input constraint 'N' in asm}}
  // arm7-error@210 {{invalid input constraint 'N' in asm}}
  // thumb1-no-error
  // thumb2-error@210 {{invalid input constraint 'N' in asm}}
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "N"(32), [ input ] "r"(i)
      :);
  // arm6-error@219 {{invalid input constraint 'N' in asm}}
  // arm7-error@219 {{invalid input constraint 'N' in asm}}
  // thumb1-error@219 {{value '32' out of range for constraint 'N'}}
  // thumb2-error@219 {{invalid input constraint 'N' in asm}}
  return res;
}

// O: Invalid (ARM/Thumb2)
//    An immediate integer which is a multiple of 4 between -508 and 508. (Thumb1)
int test_O(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "O"(1), [ input ] "r"(i)
      :);
  // arm6-error@235 {{invalid input constraint 'O' in asm}}
  // arm7-error@235 {{invalid input constraint 'O' in asm}}
  // thumb1-no-error
  // thumb2-error@235 {{invalid input constraint 'O' in asm}}
  return res;
}

// l: Same as r (ARM)
//    A low 32-bit GPR register (r0-r7). (Thumb1/Thumb2)
int test_l(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=l"(res)
      : [ constant ] "i"(10), [ input ] "l"(i)
      :); // No errors expected.
  return res;
}

// h: Invalid (ARM)
//    A high 32-bit GPR register (r8-r15). (Thumb1/Thumb2)
int test_h(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=h"(res)
      : [ constant ] "i"(10), [ input ] "h"(i)
      :);
  // arm6-error@262 {{invalid output constraint '=h' in asm}}
  // arm7-error@262 {{invalid output constraint '=h' in asm}}
  return res;
}

// s: An integer constant, but allowing only relocatable values.
int g;

int test_s(int i) {
  int res;
  __asm(
      "add %0, %1;"
      : [ result ] "=r"(res)
      : [ constant ] "s"(&g), [ input ] "r"(i)
      :); // No errors expected.
  return res;
}

// w: A 32, 64, or 128-bit floating-point/SIMD register: s0-s31, d0-d31, or q0-q15.
float test_w(float x) {
  __asm__("vsqrt.f32 %0, %1"
          : "=w"(x)
          : "w"(x)); // No error expected.
  return x;
}

// x: A 32, 64, or 128-bit floating-point/SIMD register: s0-s15, d0-d7, or q0-q3.
float test_x(float x) {
  __asm__("vsqrt.f32 %0, %1"
          : "=x"(x)
          : "x"(x)); // No error expected.
  return x;
}

// t: A 32, 64, or 128-bit floating-point/SIMD register: s0-s31, d0-d15, or q0-q7.
float test_t(float x) {
  __asm__("vsqrt.f32 %0, %1"
          : "=t"(x)
          : "t"(x)); // No error expected.
  return x;
}
