// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fno-sanitize-recover=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER
// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fsanitize-recover=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=implicit-integer-truncation -fsanitize-trap=implicit-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP

// CHECK-SANITIZE-ANYRECOVER: @[[UNSIGNED_INT:.*]] = {{.*}} c"'unsigned int'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[UNSIGNED_CHAR:.*]] = {{.*}} c"'unsigned char'\00" }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 0 }
// CHECK-SANITIZE-ANYRECOVER: @[[SIGNED_INT:.*]] = {{.*}} c"'int'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 10 }, {{.*}}* @[[SIGNED_INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 0 }
// CHECK-SANITIZE-ANYRECOVER: @[[SIGNED_CHAR:.*]] = {{.*}} c"'signed char'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 0 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 10 }, {{.*}}* @[[SIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 0 }

// CHECK-SANITIZE-ANYRECOVER: @[[UINT32:.*]] = {{.*}} c"'uint32_t' (aka 'unsigned int')\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[UINT8:.*]] = {{.*}} c"'uint8_t' (aka 'unsigned char')\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 10 }, {{.*}}* @[[UINT32]], {{.*}}* @[[UINT8]], i8 0 }

// ========================================================================== //
// The expected true-positives. These are implicit conversions, and they truncate.
// ========================================================================== //

// CHECK-LABEL: @unsigned_int_to_unsigned_char
unsigned char unsigned_int_to_unsigned_char(unsigned int src) {
  // CHECK: %[[DST:.*]] = trunc i32 %[[SRC:.*]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK: ret i8 %[[DST]]
#line 100
  return src;
}

// CHECK-LABEL: @signed_int_to_unsigned_char
unsigned char signed_int_to_unsigned_char(signed int src) {
  // CHECK: %[[DST:.*]] = trunc i32 %[[SRC:.*]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK: ret i8 %[[DST]]
#line 200
  return src;
}

// CHECK-LABEL: @unsigned_int_to_signed_char
signed char unsigned_int_to_signed_char(unsigned int src) {
  // CHECK: %[[DST:.*]] = trunc i32 %[[SRC:.*]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK: ret i8 %[[DST]]
#line 300
  return src;
}

// CHECK-LABEL: @signed_int_to_signed_char
signed char signed_int_to_signed_char(signed int src) {
  // CHECK: %[[DST:.*]] = trunc i32 %[[SRC:.*]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK: ret i8 %[[DST]]
#line 400
  return src;
}

// ========================================================================== //
// Check canonical type stuff
// ========================================================================== //

typedef unsigned int uint32_t;
typedef unsigned char uint8_t;

// CHECK-LABEL: @uint32_to_uint8
uint8_t uint32_to_uint8(uint32_t src) {
  // CHECK: %[[DST:.*]] = trunc i32 %[[SRC:.*]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_500]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_500]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.trap(){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK: ret i8 %[[DST]]
#line 500
  return src;
}

// ========================================================================== //
// Check that explicit conversion does not interfere with implicit conversion
// ========================================================================== //
// These contain one implicit truncating conversion, and one explicit truncating conversion.
// We want to make sure that we still diagnose the implicit conversion.

// Implicit truncation after explicit truncation.
// CHECK-LABEL: @explicit_conversion_interference0
unsigned char explicit_conversion_interference0(unsigned int c) {
  // CHECK-SANITIZE: %[[ANYEXT:.*]] = zext i8 %[[DST:.*]] to i16, !nosanitize
  // CHECK-SANITIZE: call
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned short)c;
}

// Implicit truncation before explicit truncation.
// CHECK-LABEL: @explicit_conversion_interference1
unsigned char explicit_conversion_interference1(unsigned int c) {
  // CHECK-SANITIZE: %[[ANYEXT:.*]] = zext i16 %[[DST:.*]] to i32, !nosanitize
  // CHECK-SANITIZE: call
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  unsigned short b;
  return (unsigned char)(b = c);
}

// ========================================================================== //
// The expected true-negatives.
// ========================================================================== //

// Sanitization is explicitly disabled.
// ========================================================================== //

// CHECK-LABEL: @blacklist_0
__attribute__((no_sanitize("undefined"))) unsigned char blacklist_0(unsigned int src) {
  // We are not in "undefined" group, so that doesn't work.
  // CHECK-SANITIZE: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @blacklist_1
__attribute__((no_sanitize("implicit-conversion"))) unsigned char blacklist_1(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @blacklist_2
__attribute__((no_sanitize("implicit-integer-truncation"))) unsigned char blacklist_2(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// Explicit truncating conversions.
// ========================================================================== //

// CHECK-LABEL: @explicit_unsigned_int_to_unsigned_char
unsigned char explicit_unsigned_int_to_unsigned_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_signed_int_to_unsigned_char
unsigned char explicit_signed_int_to_unsigned_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_unsigned_int_to_signed_char
signed char explicit_unsigned_int_to_signed_char(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// CHECK-LABEL: @explicit_signed_int_to_signed_char
signed char explicit_signed_int_to_signed_char(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// Explicit NOP conversions.
// ========================================================================== //

// CHECK-LABEL: @explicit_unsigned_int_to_unsigned_int
unsigned int explicit_unsigned_int_to_unsigned_int(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned int)src;
}

// CHECK-LABEL: @explicit_signed_int_to_signed_int
signed int explicit_signed_int_to_signed_int(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed int)src;
}

// CHECK-LABEL: @explicit_unsigned_char_to_signed_char
unsigned char explicit_unsigned_char_to_signed_char(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_signed_char_to_signed_char
signed char explicit_signed_char_to_signed_char(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// upcasts.
// ========================================================================== //

// CHECK-LABEL: @unsigned_char_to_unsigned_int
unsigned int unsigned_char_to_unsigned_int(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @signed_char_to_unsigned_int
unsigned int signed_char_to_unsigned_int(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @unsigned_char_to_signed_int
signed int unsigned_char_to_signed_int(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @signed_char_to_signed_int
signed int signed_char_to_signed_int(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// Explicit upcasts.
// ========================================================================== //

// CHECK-LABEL: @explicit_unsigned_char_to_unsigned_int
unsigned int explicit_unsigned_char_to_unsigned_int(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned int)src;
}

// CHECK-LABEL: @explicit_signed_char_to_unsigned_int
unsigned int explicit_signed_char_to_unsigned_int(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned int)src;
}

// CHECK-LABEL: @explicit_unsigned_char_to_signed_int
signed int explicit_unsigned_char_to_signed_int(unsigned char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed int)src;
}

// CHECK-LABEL: @explicit_signed_char_to_signed_int
signed int explicit_signed_char_to_signed_int(signed char src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed int)src;
}

// conversions to to boolean type are not counted as truncation.
// ========================================================================== //

// CHECK-LABEL: @unsigned_int_to_bool
_Bool unsigned_int_to_bool(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @signed_int_to_bool
_Bool signed_int_to_bool(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @explicit_unsigned_int_to_bool
_Bool explicit_unsigned_int_to_bool(unsigned int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (_Bool)src;
}

// CHECK-LABEL: @explicit_signed_int_to_bool
_Bool explicit_signed_int_to_bool(signed int src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (_Bool)src;
}

// Explicit truncating conversions from pointer to a much-smaller integer.
// Can not have an implicit conversion from pointer to an integer.
// Can not have an implicit conversion between two enums.
// ========================================================================== //

// CHECK-LABEL: @explicit_voidptr_to_unsigned_char
unsigned char explicit_voidptr_to_unsigned_char(void *src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (unsigned char)src;
}

// CHECK-LABEL: @explicit_voidptr_to_signed_char
signed char explicit_voidptr_to_signed_char(void *src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return (signed char)src;
}

// Implicit truncating conversions from floating-point may result in precision loss.
// ========================================================================== //

// CHECK-LABEL: @float_to_unsigned_int
unsigned int float_to_unsigned_int(float src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @float_to_signed_int
signed int float_to_signed_int(float src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @double_to_unsigned_int
unsigned int double_to_unsigned_int(double src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// CHECK-LABEL: @double_to_signed_int
signed int double_to_signed_int(double src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}

// Implicit truncating conversions between fp may result in precision loss.
// ========================================================================== //

// CHECK-LABEL: @double_to_float
float double_to_float(double src) {
  // CHECK-SANITIZE-NOT: call
  // CHECK: }
  return src;
}
