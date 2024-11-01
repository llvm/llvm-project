// RUN: %check_clang_tidy -std=cl2.0 %s readability-uppercase-literal-suffix %t -- -- -target x86_64-pc-linux-gnu -I %S -x cl
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -fix -- -target x86_64-pc-linux-gnu -I %S -std=cl2.0 -x cl
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -warnings-as-errors='-*,readability-uppercase-literal-suffix' -- -target x86_64-pc-linux-gnu -I %S -std=cl2.0 -x cl

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

void floating_point_half_suffix() {
  static half v0 = 0x0p0; // no literal

  // half

  static half v2 = 1.h;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: floating point literal has suffix 'h', which is not uppercase
  // CHECK-MESSAGES-NEXT: static half v2 = 1.h;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: H{{$}}
  // CHECK-HIXES: static half v2 = 1.H;

  static half v3 = 1.e0h;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: floating point literal has suffix 'h', which is not uppercase
  // CHECK-MESSAGES-NEXT: static half v3 = 1.e0h;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: H{{$}}
  // CHECK-HIXES: static half v3 = 1.e0H;

  static half v4 = 1.H; // OK.

  static half v5 = 1.e0H; // OK.
}
