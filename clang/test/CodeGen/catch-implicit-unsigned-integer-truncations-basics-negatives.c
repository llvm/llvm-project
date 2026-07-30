// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation -fsanitize-recover=implicit-unsigned-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// CHECK-DAG: @[[LINE_100_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}, {{.*}}, i8 1, i32 0 }
// CHECK-DAG: @[[LINE_200_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 200, i32 10 }, {{.*}}, {{.*}}, i8 1, i32 0 }

// CHECK-LABEL: @ignorelist_0_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("undefined"))) unsigned char ignorelist_0_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_100_UNSIGNED_TRUNCATION]]
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
  // CHECK: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_200_UNSIGNED_TRUNCATION]]
#line 200
  return x;
}
