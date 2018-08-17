// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fsanitize-recover=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK

// Test plan:
//  * Two types - int and char
//  * Two signs - signed and unsigned
//  * Square that - we have input and output types.
// Thus, there are total of (2*2)^2 == 16 tests.
// These are all the possible variations/combinations of casts.
// However, not all of them should result in the check.
// So here, we *only* check which should and which should not result in checks.

// CHECK-DAG: @[[LINE_500_TRUNCATION:.*]] = {{.*}}, i32 500, i32 10 }, {{.*}}, {{.*}}, i8 0 }
// CHECK-DAG: @[[LINE_1100_TRUNCATION:.*]] = {{.*}}, i32 1100, i32 10 }, {{.*}}, {{.*}}, i8 0 }
// CHECK-DAG: @[[LINE_1500_TRUNCATION:.*]] = {{.*}}, i32 1500, i32 10 }, {{.*}}, {{.*}}, i8 0 }
// CHECK-DAG: @[[LINE_1600_TRUNCATION:.*]] = {{.*}}, i32 1600, i32 10 }, {{.*}}, {{.*}}, i8 0 }

// CHECK-LABEL: @convert_unsigned_int_to_unsigned_int
unsigned int convert_unsigned_int_to_unsigned_int(unsigned int x) {
  // CHECK: }
#line 100
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_unsigned_char
unsigned char convert_unsigned_char_to_unsigned_char(unsigned char x) {
  // CHECK: }
#line 200
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_signed_int
signed int convert_signed_int_to_signed_int(signed int x) {
  // CHECK: }
#line 300
  return x;
}

// CHECK-LABEL: @convert_signed_char_to_signed_char
signed char convert_signed_char_to_signed_char(signed char x) {
  // CHECK: }
#line 400
  return x;
}

// CHECK-LABEL: @convert_unsigned_int_to_unsigned_char
unsigned char convert_unsigned_int_to_unsigned_char(unsigned int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_500_TRUNCATION]] to i8*)
  // CHECK: }
#line 500
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_unsigned_int
unsigned int convert_unsigned_char_to_unsigned_int(unsigned char x) {
  // CHECK: }
#line 600
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_signed_int
signed int convert_unsigned_char_to_signed_int(unsigned char x) {
  // CHECK: }
#line 700
  return x;
}

// CHECK-LABEL: @convert_signed_char_to_signed_int
signed int convert_signed_char_to_signed_int(signed char x) {
  // CHECK: }
#line 800
  return x;
}

// CHECK-LABEL: @convert_unsigned_int_to_signed_int
signed int convert_unsigned_int_to_signed_int(unsigned int x) {
  // CHECK: }
#line 900
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_unsigned_int
unsigned int convert_signed_int_to_unsigned_int(signed int x) {
  // CHECK: }
#line 1000
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_unsigned_char
unsigned char convert_signed_int_to_unsigned_char(signed int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1100_TRUNCATION]] to i8*)
  // CHECK: }
#line 1100
  return x;
}

// CHECK-LABEL: @convert_signed_char_to_unsigned_char
unsigned char convert_signed_char_to_unsigned_char(signed char x) {
  // CHECK: }
#line 1200
  return x;
}

// CHECK-LABEL: @convert_unsigned_char_to_signed_char
signed char convert_unsigned_char_to_signed_char(unsigned char x) {
  // CHECK: }
#line 1300
  return x;
}

// CHECK-LABEL: @convert_signed_char_to_unsigned_int
unsigned int convert_signed_char_to_unsigned_int(signed char x) {
  // CHECK: }
#line 1400
  return x;
}

// CHECK-LABEL: @convert_unsigned_int_to_signed_char
signed char convert_unsigned_int_to_signed_char(unsigned int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1500_TRUNCATION]] to i8*)
  // CHECK: }
#line 1500
  return x;
}

// CHECK-LABEL: @convert_signed_int_to_signed_char
signed char convert_signed_int_to_signed_char(signed int x) {
  // CHECK: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_1600_TRUNCATION]] to i8*)
  // CHECK: }
#line 1600
  return x;
}
