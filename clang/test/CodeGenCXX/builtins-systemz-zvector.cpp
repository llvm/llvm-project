// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-linux-gnu \
// RUN: -fzvector -fno-lax-vector-conversions -std=c++11 \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s

bool gb;

// There was an issue where we weren't properly converting constexprs to
// vectors with elements of the appropriate width. (e.g.
// (vector signed short)0 would be lowered as [4 x i32] in some cases)

// CHECK-LABEL: @_Z8testIntsDv4_i
void testInts(vector int VI) {
  constexpr vector int CI1 = (vector int)0LL;
  // CHECK: icmp
  gb = (VI == CI1)[0];

  // Likewise for float inits.
  constexpr vector int CI2 = (vector int)char(0);
  // CHECK: icmp
  gb = (VI == CI2)[0];

  constexpr vector int CF1 = (vector int)0.0;
  // CHECK: icmp
  gb = (VI == CF1)[0];

  constexpr vector int CF2 = (vector int)0.0f;
  // CHECK: icmp
  gb = (VI == CF2)[0];
}

// CHECK-LABEL: @_Z10testFloatsDv2_d
void testFloats(vector double VD) {
  constexpr vector double CI1 = (vector double)0LL;
  // CHECK: fcmp
  gb = (VD == CI1)[0];

  // Likewise for float inits.
  constexpr vector double CI2 = (vector double)char(0);
  // CHECK: fcmp
  gb = (VD == CI2)[0];

  constexpr vector double CF1 = (vector double)0.0;
  // CHECK: fcmp
  gb = (VD == CF1)[0];

  constexpr vector double CF2 = (vector double)0.0f;
  // CHECK: fcmp
  gb = (VD == CF2)[0];
}
