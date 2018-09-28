// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fsanitize-recover=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// CHECK-DAG: @[[LINE_100_TRUNCATION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}, {{.*}}, i8 0 }
// CHECK-DAG: @[[LINE_300_TRUNCATION:.*]] = {{.*}}, i32 300, i32 10 }, {{.*}}, {{.*}}, i8 0 }

//----------------------------------------------------------------------------//
// Unsigned case.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @blacklist_0_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("undefined"))) unsigned char blacklist_0_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100_TRUNCATION]] to i8*)
#line 100
  return x;
}

// CHECK-LABEL: @blacklist_1_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("integer"))) unsigned char blacklist_1_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @blacklist_2_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-conversion"))) unsigned char blacklist_2_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @blacklist_3_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-integer-truncation"))) unsigned char blacklist_3_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

//----------------------------------------------------------------------------//
// Signed case.
//----------------------------------------------------------------------------//

// CHECK-LABEL: @blacklist_0_convert_signed_int_to_signed_char
__attribute__((no_sanitize("undefined"))) signed char blacklist_0_convert_signed_int_to_signed_char(signed int x) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300_TRUNCATION]] to i8*)
#line 300
  return x;
}

// CHECK-LABEL: @blacklist_1_convert_signed_int_to_signed_char
__attribute__((no_sanitize("integer"))) signed char blacklist_1_convert_signed_int_to_signed_char(signed int x) {
  return x;
}

// CHECK-LABEL: @blacklist_2_convert_signed_int_to_signed_char
__attribute__((no_sanitize("implicit-conversion"))) signed char blacklist_2_convert_signed_int_to_signed_char(signed int x) {
  return x;
}

// CHECK-LABEL: @blacklist_3_convert_signed_int_to_signed_char
__attribute__((no_sanitize("implicit-integer-truncation"))) signed char blacklist_3_convert_signed_int_to_signed_char(signed int x) {
  return x;
}
