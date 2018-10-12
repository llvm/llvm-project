// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation -fsanitize-recover=implicit-unsigned-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// CHECK-DAG: @[[LINE_100_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}, {{.*}}, i8 1 }
// CHECK-DAG: @[[LINE_200_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 200, i32 10 }, {{.*}}, {{.*}}, i8 1 }

// CHECK-LABEL: @blacklist_0_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("undefined"))) unsigned char blacklist_0_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100_UNSIGNED_TRUNCATION]] to i8*)
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

// CHECK-LABEL: @blacklist_4_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-unsigned-integer-truncation"))) unsigned char blacklist_4_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  return x;
}

// CHECK-LABEL: @blacklist_5_convert_unsigned_int_to_unsigned_char
__attribute__((no_sanitize("implicit-signed-integer-truncation"))) unsigned char blacklist_5_convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // This is an unsigned truncation, not signed-one.
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200_UNSIGNED_TRUNCATION]] to i8*)
#line 200
  return x;
}
