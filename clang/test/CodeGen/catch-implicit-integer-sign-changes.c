// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fno-sanitize-recover=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fsanitize-recover=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=implicit-integer-sign-change -fsanitize-trap=implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[UNSIGNED_INT:.*]] = {{.*}} c"'unsigned int'\00" }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[SIGNED_INT:.*]] = {{.*}} c"'int'\00" }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_100:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_INT]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_200:.*]] = {{.*}}, i32 200, i32 10 }, {{.*}}* @[[SIGNED_INT]], {{.*}}* @[[UNSIGNED_INT]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[UNSIGNED_CHAR:.*]] = {{.*}} c"'unsigned char'\00" }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_300:.*]] = {{.*}}, i32 300, i32 10 }, {{.*}}* @[[SIGNED_INT]], {{.*}}* @[[UNSIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[SIGNED_CHAR:.*]] = {{.*}} c"'signed char'\00" }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_400:.*]] = {{.*}}, i32 400, i32 10 }, {{.*}}* @[[SIGNED_CHAR]], {{.*}}* @[[UNSIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_500:.*]] = {{.*}}, i32 500, i32 10 }, {{.*}}* @[[UNSIGNED_CHAR]], {{.*}}* @[[SIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_600:.*]] = {{.*}}, i32 600, i32 10 }, {{.*}}* @[[SIGNED_CHAR]], {{.*}}* @[[UNSIGNED_INT]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_700:.*]] = {{.*}}, i32 700, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_800:.*]] = {{.*}}, i32 800, i32 10 }, {{.*}}* @[[SIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER: @[[UINT32:.*]] = {{.*}} c"'uint32_t' (aka 'unsigned int')\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[INT32:.*]] = {{.*}} c"'int32_t' (aka 'int')\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_900:.*]] = {{.*}}, i32 900, i32 10 }, {{.*}}* @[[UINT32]], {{.*}}* @[[INT32]], i8 3 }

// ========================================================================== //
// The expected true-positives.
// These are implicit, potentially sign-altering, conversions.
// ========================================================================== //

// These 3 result (after optimizations) in simple 'icmp sge i32 %src, 0'.

// CHECK-LABEL: @unsigned_int_to_signed_int
// CHECK-SAME: (i32 %[[SRC:.*]])
signed int unsigned_int_to_signed_int(unsigned int src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i32 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i32 %[[DST]]
  // CHECK-NEXT: }
#line 100
  return src;
}

// CHECK-LABEL: @signed_int_to_unsigned_int
// CHECK-SAME: (i32 %[[SRC:.*]])
unsigned int signed_int_to_unsigned_int(signed int src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-SANITIZE-NEXT: %[[SRC_NEGATIVITYCHECK:.*]] = icmp slt i32 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 %[[SRC_NEGATIVITYCHECK]], false, !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i32 %[[DST]]
  // CHECK-NEXT: }
#line 200
  return src;
}

// CHECK-LABEL: @signed_int_to_unsigned_char
// CHECK-SAME: (i32 %[[SRC:.*]])
unsigned char signed_int_to_unsigned_char(signed int src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[SRC_NEGATIVITYCHECK:.*]] = icmp slt i32 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 %[[SRC_NEGATIVITYCHECK]], false, !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 300
  return src;
}

// These 3 result (after optimizations) in simple 'icmp sge i8 %src, 0'

// CHECK-LABEL: @signed_char_to_unsigned_char
// CHECK-SAME: (i8 signext %[[SRC:.*]])
unsigned char signed_char_to_unsigned_char(signed char src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i8
  // CHECK-NEXT: store i8 %[[SRC]], i8* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i8, i8* %[[SRC_ADDR]]
  // CHECK-SANITIZE-NEXT: %[[SRC_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 %[[SRC_NEGATIVITYCHECK]], false, !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[DST]]
  // CHECK-NEXT: }
#line 400
  return src;
}

// CHECK-LABEL: @unsigned_char_to_signed_char
// CHECK-SAME: (i8 zeroext %[[SRC:.*]])
signed char unsigned_char_to_signed_char(unsigned char src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i8
  // CHECK-NEXT: store i8 %[[SRC]], i8* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i8, i8* %[[SRC_ADDR]]
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_500]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_500]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[DST]]
  // CHECK-NEXT: }
#line 500
  return src;
}

// CHECK-LABEL: @signed_char_to_unsigned_int
// CHECK-SAME: (i8 signext %[[SRC:.*]])
unsigned int signed_char_to_unsigned_int(signed char src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i8
  // CHECK-NEXT: store i8 %[[SRC]], i8* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i8, i8* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = sext i8 %[[DST]] to i32
  // CHECK-SANITIZE-NEXT: %[[SRC_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 %[[SRC_NEGATIVITYCHECK]], false, !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i32 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_600]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_600]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i32 %[[CONV]]
  // CHECK-NEXT: }
#line 600
  return src;
}

// This one result (after optimizations) in 'icmp sge i8 (trunc i32 %src), 0'

// CHECK-LABEL: @unsigned_int_to_signed_char
// CHECK-SAME: (i32 %[[SRC:.*]])
signed char unsigned_int_to_signed_char(unsigned int src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[CONV]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_700]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_700]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 700
  return src;
}

// The worst one: 'xor i1 (icmp sge i8 (trunc i32 %x), 0), (icmp sge i32 %x, 0)'

// CHECK-LABEL: @signed_int_to_signed_char
// CHECK-SAME: (i32 %[[SRC:.*]])
signed char signed_int_to_signed_char(signed int x) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[SRC_NEGATIVITYCHECK:.*]] = icmp slt i32 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[CONV]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 %[[SRC_NEGATIVITYCHECK]], %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_800]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_800]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 800
  return x;
}

// ========================================================================== //
// Check canonical type stuff
// ========================================================================== //

typedef unsigned int uint32_t;
typedef signed int int32_t;

// CHECK-LABEL: @uint32_t_to_int32_t
// CHECK-SAME: (i32 %[[SRC:.*]])
int32_t uint32_t_to_int32_t(uint32_t src) {
  // CHECK: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i32 %[[DST]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_900]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_900]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i32 %[[DST]]
  // CHECK-NEXT: }
#line 900
  return src;
}

// ========================================================================== //
// Check that explicit conversion does not interfere with implicit conversion
// ========================================================================== //
// These contain one implicit and one explicit sign-changing conversion.
// We want to make sure that we still diagnose the implicit conversion.

// Implicit sign-change after explicit sign-change.
// CHECK-LABEL: @explicit_conversion_interference0
unsigned int explicit_conversion_interference0(unsigned int c) {
  // CHECK-SANITIZE: call
  return (signed int)c;
}

// Implicit sign-change before explicit sign-change.
// CHECK-LABEL: @explicit_conversion_interference1
unsigned int explicit_conversion_interference1(unsigned int c) {
  // CHECK-SANITIZE: call
  signed int b;
  return (unsigned int)(b = c);
}
