// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fsanitize-recover=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// Test plan:
//  * Two types - int and char
//  * Two signs - signed and unsigned
//  * Square that - we have input and output types.
// Thus, there are total of (2*2)^2 == 16 tests.
// These are all the possible variations/combinations of casts.
// However, not all of them should result in the check.
// So here, we *only* check which should and which should not result in checks.

// CHECK-DAG: @[[LINE_900_SIGN_CHANGE:.*]] = {{.*}}, i32 900, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1000_SIGN_CHANGE:.*]] = {{.*}}, i32 1000, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1100_SIGN_CHANGE:.*]] = {{.*}}, i32 1100, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1200_SIGN_CHANGE:.*]] = {{.*}}, i32 1200, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1300_SIGN_CHANGE:.*]] = {{.*}}, i32 1300, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1400_SIGN_CHANGE:.*]] = {{.*}}, i32 1400, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1500_SIGN_CHANGE:.*]] = {{.*}}, i32 1500, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }
// CHECK-DAG: @[[LINE_1600_SIGN_CHANGE:.*]] = {{.*}}, i32 1600, i32 10 }, {{.*}}, {{.*}}, i8 3, i32 0 }

//============================================================================//
// Half of the cases do not need the check.                                   //
//============================================================================//

//----------------------------------------------------------------------------//
// No cast happens at all. No check needed.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @convert_unsigned_int_to_unsigned_int
unsigned int convert_unsigned_int_to_unsigned_int(unsigned int x) {
#line 100
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_unsigned_char
unsigned char convert_unsigned_char_to_unsigned_char(unsigned char x) {
#line 200
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_signed_int
signed int convert_signed_int_to_signed_int(signed int x) {
#line 300
  return x;
}

// CHECK-LABEL: @convert_signed_char_to_signed_char
signed char convert_signed_char_to_signed_char(signed char x) {
#line 400
  return x;
}

//----------------------------------------------------------------------------//
// Both types are unsigned. No check needed.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @convert_unsigned_int_to_unsigned_char
unsigned char convert_unsigned_int_to_unsigned_char(unsigned int x) {
#line 500
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_unsigned_int
unsigned int convert_unsigned_char_to_unsigned_int(unsigned char x) {
#line 600
  return x;
}

//----------------------------------------------------------------------------//
// Source type was unsigned, destination type is signed, but non-negative.
// Because zero-extension happens - the sign bit will be 0. No check needed.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @convert_unsigned_char_to_signed_int
signed int convert_unsigned_char_to_signed_int(unsigned char x) {
#line 700
  return x;
}

//----------------------------------------------------------------------------//
// Both types are signed, and have the same sign, since sign-extension happens,
// i.e. the sign bit will be propagated. No check needed.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @convert_signed_char_to_signed_int
signed int convert_signed_char_to_signed_int(signed char x) {
#line 800
  return x;
}

//============================================================================//
// The remaining 8 cases *do* need the check.                                 //
//============================================================================//

// These 3 result in simple 'icmp sge i32 %x, 0'

// CHECK-LABEL: @convert_unsigned_int_to_signed_int
signed int convert_unsigned_int_to_signed_int(unsigned int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_900_SIGN_CHANGE]]
#line 900
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_unsigned_int
unsigned int convert_signed_int_to_unsigned_int(signed int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1000_SIGN_CHANGE]]
#line 1000
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_unsigned_char
unsigned char convert_signed_int_to_unsigned_char(signed int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1100_SIGN_CHANGE]]
#line 1100
  return x;
}

// These 3 result in simple 'icmp sge i8 %x, 0'

// CHECK-LABEL: @convert_signed_char_to_unsigned_char
unsigned char convert_signed_char_to_unsigned_char(signed char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1200_SIGN_CHANGE]]
#line 1200
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_signed_char
signed char convert_unsigned_char_to_signed_char(unsigned char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1300_SIGN_CHANGE]]
#line 1300
  return x;
}

// CHECK-LABEL: @convert_signed_char_to_unsigned_int
unsigned int convert_signed_char_to_unsigned_int(signed char x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1400_SIGN_CHANGE]]
#line 1400
  return x;
}

// 'icmp sge i8 (trunc i32 %x), 0'

// CHECK-LABEL: @convert_unsigned_int_to_signed_char
signed char convert_unsigned_int_to_signed_char(unsigned int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1500_SIGN_CHANGE]]
#line 1500
  return x;
}

// 'xor i1 (icmp sge i8 (trunc i32 %x), 0), (icmp sge i32 %x, 0)'

// CHECK-LABEL: @convert_signed_int_to_signed_char
signed char convert_signed_int_to_signed_char(signed int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1600_SIGN_CHANGE]]
#line 1600
  return x;
}
