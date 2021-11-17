// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -fsanitize-recover=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// CHECK-DAG: @[[LINE_100_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}, {{.*}}, i8 1 }
// CHECK-DAG: @[[LINE_200_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 200, i32 10 }, {{.*}}, {{.*}}, i8 1 }
// CHECK-DAG: @[[LINE_300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 300, i32 10 }, {{.*}}, {{.*}}, i8 2 }
// CHECK-DAG: @[[LINE_400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 400, i32 10 }, {{.*}}, {{.*}}, i8 2 }

//----------------------------------------------------------------------------//
// Unsigned case.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @ignorelist_0_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("undefined"))) unsigned char ignorelist_0_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100_UNSIGNED_TRUNCATION]] to i8*)
#line 100
  return x;
}

// CHECK-LABEL: @ignorelist_1_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("integer"))) unsigned char ignorelist_1_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_2_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-conversion"))) unsigned char ignorelist_2_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_3_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-integer-truncation"))) unsigned char ignorelist_3_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_4_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-unsigned-integer-truncation"))) unsigned char ignorelist_4_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_5_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-signed-integer-truncation"))) unsigned char ignorelist_5_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // This is an unsigned truncation, not signed-one.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200_UNSIGNED_TRUNCATION]] to i8*)
#line 200
  return x;
}

//----------------------------------------------------------------------------//
// Signed case.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @ignorelist_0_convert_signed_int_to_signed_char
__attribute__((no_sanitize("undefined"))) signed char ignorelist_0_convert_signed_int_to_signed_char(signed int x) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300_SIGNED_TRUNCATION]] to i8*)
#line 300
  return x;
}

// CHECK-LABEL: @ignorelist_1_convert_signed_int_to_signed_char
__attribute__((no_sanitize("integer"))) signed char ignorelist_1_convert_signed_int_to_signed_char(signed int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_2_convert_signed_int_to_signed_char
__attribute__((no_sanitize("implicit-conversion"))) signed char ignorelist_2_convert_signed_int_to_signed_char(signed int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_3_convert_signed_int_to_signed_char
__attribute__((no_sanitize("implicit-integer-truncation"))) signed char ignorelist_3_convert_signed_int_to_signed_char(signed int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_4_convert_signed_int_to_signed_char
__attribute__((no_sanitize("implicit-signed-integer-truncation"))) signed char ignorelist_4_convert_signed_int_to_signed_char(signed int x) {
  return x;
}

// CHECK-LABEL: @ignorelist_5_convert_signed_int_to_signed_char
__attribute__((no_sanitize("implicit-unsigned-integer-truncation"))) signed char ignorelist_5_convert_signed_int_to_signed_char(signed int x) {
  // This is an signed truncation, not unsigned-one.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400_SIGNED_TRUNCATION]] to i8*)
#line 400
  return x;
}
