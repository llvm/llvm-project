// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fno-sanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-recover=implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-trap=implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// CHECK-SANITIZE-ANYRECOVER: @[[UNSIGNED_INT:.*]] = {{.*}} c"'unsigned int'\00" }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[SIGNED_CHAR:.*]] = {{.*}} c"'signed char'\00" }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_100_SIGNED_TRUNCATION_OR_SIGN_CHANGE:.*]] = {{.*}}, i32 100, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 4 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_200_SIGN_CHANGE:.*]] = {{.*}}, i32 200, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_300_SIGN_CHANGE:.*]] = {{.*}}, i32 300, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 3 }
// CHECK-SANITIZE-ANYRECOVER-NEXT: @[[LINE_400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 400, i32 10 }, {{.*}}* @[[UNSIGNED_INT]], {{.*}}* @[[SIGNED_CHAR]], i8 2 }

//============================================================================//
// Both sanitizers are enabled, and not disabled per-function.
//============================================================================//

// CHECK-LABEL: @unsigned_int_to_signed_char
// CHECK-SAME: i32 noundef %[[SRC:.*]])
signed char unsigned_int_to_signed_char(unsigned int src) {
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-NEXT: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[CONV]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[CONV]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[DST]], !nosanitize
  // CHECK-SANITIZE-NEXT: %[[BOTHCHECKS:.*]] = and i1 %[[SIGNCHANGECHECK]], %[[TRUNCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[BOTHCHECKS]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100_SIGNED_TRUNCATION_OR_SIGN_CHANGE]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_100_SIGNED_TRUNCATION_OR_SIGN_CHANGE]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 100
  return src;
}

//============================================================================//
// Truncation sanitizer is disabled per-function.
//============================================================================//

// CHECK-LABEL: @unsigned_int_to_signed_char__no_truncation_sanitizer
// CHECK-SAME: i32 noundef %[[SRC:.*]])
__attribute__((no_sanitize("implicit-integer-truncation"))) signed char
unsigned_int_to_signed_char__no_truncation_sanitizer(unsigned int src) {
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-NEXT: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[CONV]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200_SIGN_CHANGE]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_200_SIGN_CHANGE]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 200
  return src;
}

//============================================================================//
// Signed truncation sanitizer is disabled per-function.
//============================================================================//

// CHECK-LABEL: @unsigned_int_to_signed_char__no_signed_truncation_sanitizer
// CHECK-SAME: i32 noundef %[[SRC:.*]])
__attribute__((no_sanitize("implicit-signed-integer-truncation"))) signed char
unsigned_int_to_signed_char__no_signed_truncation_sanitizer(unsigned int src) {
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-NEXT: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[DST_NEGATIVITYCHECK:.*]] = icmp slt i8 %[[CONV]], 0, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[SIGNCHANGECHECK:.*]] = icmp eq i1 false, %[[DST_NEGATIVITYCHECK]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[SIGNCHANGECHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300_SIGN_CHANGE]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_300_SIGN_CHANGE]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 300
  return src;
}

//============================================================================//
// Sign change sanitizer is disabled per-function
//============================================================================//

// CHECK-LABEL: @unsigned_int_to_signed_char__no_sign_change_sanitizer
// CHECK-SAME: i32 noundef %[[SRC:.*]])
__attribute__((no_sanitize("implicit-integer-sign-change"))) signed char
unsigned_int_to_signed_char__no_sign_change_sanitizer(unsigned int src) {
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-NEXT: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[CONV]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[DST]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTCONV:.*]] = zext i8 %[[CONV]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400_SIGNED_TRUNCATION]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(i8* bitcast ({ {{{.*}}}, {{{.*}}}*, {{{.*}}}*, i8 }* @[[LINE_400_SIGNED_TRUNCATION]] to i8*), i64 %[[EXTSRC]], i64 %[[EXTCONV]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
#line 400
  return src;
}

//============================================================================//
// Both sanitizers are disabled per-function.
//============================================================================//

// CHECK-LABEL: @unsigned_int_to_signed_char__no_sanitizers
// CHECK-SAME: i32 noundef %[[SRC:.*]])
__attribute__((no_sanitize("implicit-integer-truncation"),
               no_sanitize("implicit-integer-sign-change"))) signed char
unsigned_int_to_signed_char__no_sanitizers(unsigned int src) {
  // CHECK-NEXT: [[ENTRY:.*]]:
  // CHECK-NEXT: %[[SRC_ADDR:.*]] = alloca i32
  // CHECK-NEXT: store i32 %[[SRC]], i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[DST:.*]] = load i32, i32* %[[SRC_ADDR]]
  // CHECK-NEXT: %[[CONV:.*]] = trunc i32 %[[DST]] to i8
  // CHECK-NEXT: ret i8 %[[CONV]]
  // CHECK-NEXT: }
  return src;
}
