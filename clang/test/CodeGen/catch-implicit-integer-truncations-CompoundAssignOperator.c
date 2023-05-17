// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s --check-prefix=CHECK

// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -fno-sanitize-recover=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-NORECOVER,CHECK-SANITIZE-UNREACHABLE
// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -fsanitize-recover=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-ANYRECOVER,CHECK-SANITIZE-RECOVER
// RUN: %clang_cc1 -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -fsanitize-trap=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_implicit_conversion" --check-prefixes=CHECK,CHECK-SANITIZE,CHECK-SANITIZE-TRAP,CHECK-SANITIZE-UNREACHABLE

// LHS can be of 2 types: unsigned char and signed char
// RHS can be of 4 types: unsigned char, signed char, unsigned int, signed int.
// Therefore there are total of 8 tests per group.

// Also there are total of 10 compound operators (+=, -=, *=, /=, %=, <<=, >>=, &=, ^=, |=)

// CHECK-SANITIZE-ANYRECOVER: @[[INT:.*]] = {{.*}} c"'int'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[UNSIGNED_CHAR:.*]] = {{.*}} c"'unsigned char'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 100, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 200, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[UNSIGNED_INT:.*]] = {{.*}} c"'unsigned int'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_300_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 300, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 400, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[SIGNED_CHAR:.*]] = {{.*}} c"'signed char'\00" }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_500_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 500, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 600, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_700_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 700, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 800, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_900_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 900, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1000, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1100_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 1100, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1200, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1300, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1400, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1500_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1500, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1600, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1700_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1700, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 1800, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_1900_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 1900, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2000, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2100, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2200, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2300, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2400, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2500_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2500, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2600, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2700_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 2700, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2800, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_2900_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 2900, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3000, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3100, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3200, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3300, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3400, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3500_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 3500, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3600, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3700_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3700, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3800, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_3900_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 3900, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4000, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4100, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4200, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4300, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4400, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4500_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4500, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4600, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4700_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4700, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4800, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_4900_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 4900, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5000, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5100, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5200, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5300, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5400, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5500_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5500, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5600, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5700_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5700, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 5800, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_5900_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 5900, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6000, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6100, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6200, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6300, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6400, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6500_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6500, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6600, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6700_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 6700, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6800, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_6900_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 6900, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7000, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7100_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7100, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7200_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7200, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7300_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7300, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7400_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7400, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7500_UNSIGNED_TRUNCATION:.*]] = {{.*}}, i32 7500, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[UNSIGNED_CHAR]], i8 1 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7600_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7600, i32 10 }, ptr @[[INT]], ptr @[[UNSIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7700_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7700, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7800_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7800, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_7900_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 7900, i32 10 }, ptr @[[UNSIGNED_INT]], ptr @[[SIGNED_CHAR]], i8 2 }
// CHECK-SANITIZE-ANYRECOVER: @[[LINE_8000_SIGNED_TRUNCATION:.*]] = {{.*}}, i32 8000, i32 10 }, ptr @[[INT]], ptr @[[SIGNED_CHAR]], i8 2 }

//----------------------------------------------------------------------------//
// Compound add operator.                                                     //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_add_signed_char_unsigned_char
void unsigned_char_add_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 100
  (*LHS) += RHS;
}

// CHECK-LABEL: @unsigned_char_add_signed_char_signed_char
void unsigned_char_add_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 200
  (*LHS) += RHS;
}

// CHECK-LABEL: @unsigned_char_add_signed_char_unsigned_int
void unsigned_char_add_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_300_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_300_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 300
  (*LHS) += RHS;
}

// CHECK-LABEL: @unsigned_char_add_signed_char_signed_int
void unsigned_char_add_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add nsw i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 400
  (*LHS) += RHS;
}

// CHECK-LABEL: @signed_char_add_unsigned_char
void signed_char_add_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 500
  (*LHS) += RHS;
}

// CHECK-LABEL: @signed_char_add_signed_char
void signed_char_add_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 600
  (*LHS) += RHS;
}

// CHECK-LABEL: @signed_char_add_signed_char_unsigned_int
void signed_char_add_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 700
  (*LHS) += RHS;
}

// CHECK-LABEL: @signed_char_add_signed_char_signed_int
void signed_char_add_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = add nsw i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 800
  (*LHS) += RHS;
}

//----------------------------------------------------------------------------//
// Compound subtract operator.                                                //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_sub_signed_char_unsigned_char
void unsigned_char_sub_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 900
  (*LHS) -= RHS;
}

// CHECK-LABEL: @unsigned_char_sub_signed_char_signed_char
void unsigned_char_sub_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1000
  (*LHS) -= RHS;
}

// CHECK-LABEL: @unsigned_char_sub_signed_char_unsigned_int
void unsigned_char_sub_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1100_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1100_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1100
  (*LHS) -= RHS;
}

// CHECK-LABEL: @unsigned_char_sub_signed_char_signed_int
void unsigned_char_sub_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub nsw i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1200
  (*LHS) -= RHS;
}

// CHECK-LABEL: @signed_char_sub_unsigned_char
void signed_char_sub_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1300
  (*LHS) -= RHS;
}

// CHECK-LABEL: @signed_char_sub_signed_char
void signed_char_sub_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1400
  (*LHS) -= RHS;
}

// CHECK-LABEL: @signed_char_sub_signed_char_unsigned_int
void signed_char_sub_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1500
  (*LHS) -= RHS;
}

// CHECK-LABEL: @signed_char_sub_signed_char_signed_int
void signed_char_sub_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sub nsw i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1600
  (*LHS) -= RHS;
}

//----------------------------------------------------------------------------//
// Compound multiply operator.                                                //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_mul_signed_char_unsigned_char
void unsigned_char_mul_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1700
  (*LHS) *= RHS;
}

// CHECK-LABEL: @unsigned_char_mul_signed_char_signed_char
void unsigned_char_mul_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1800
  (*LHS) *= RHS;
}

// CHECK-LABEL: @unsigned_char_mul_signed_char_unsigned_int
void unsigned_char_mul_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_1900_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_1900_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 1900
  (*LHS) *= RHS;
}

// CHECK-LABEL: @unsigned_char_mul_signed_char_signed_int
void unsigned_char_mul_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul nsw i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2000
  (*LHS) *= RHS;
}

// CHECK-LABEL: @signed_char_mul_unsigned_char
void signed_char_mul_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2100
  (*LHS) *= RHS;
}

// CHECK-LABEL: @signed_char_mul_signed_char
void signed_char_mul_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul nsw i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2200
  (*LHS) *= RHS;
}

// CHECK-LABEL: @signed_char_mul_signed_char_unsigned_int
void signed_char_mul_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2300
  (*LHS) *= RHS;
}

// CHECK-LABEL: @signed_char_mul_signed_char_signed_int
void signed_char_mul_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = mul nsw i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2400
  (*LHS) *= RHS;
}

//----------------------------------------------------------------------------//
// Compound divide operator.                                                  //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_div_signed_char_unsigned_char
void unsigned_char_div_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sdiv i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2500
  (*LHS) /= RHS;
}

// CHECK-LABEL: @unsigned_char_div_signed_char_signed_char
void unsigned_char_div_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sdiv i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2600
  (*LHS) /= RHS;
}

// CHECK-LABEL: @unsigned_char_div_signed_char_unsigned_int
void unsigned_char_div_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = udiv i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2700_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2700_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2700
  (*LHS) /= RHS;
}

// CHECK-LABEL: @unsigned_char_div_signed_char_signed_int
void unsigned_char_div_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sdiv i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2800
  (*LHS) /= RHS;
}

// CHECK-LABEL: @signed_char_div_unsigned_char
void signed_char_div_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sdiv i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_2900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_2900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 2900
  (*LHS) /= RHS;
}

// CHECK-LABEL: @signed_char_div_signed_char
void signed_char_div_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sdiv i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3000
  (*LHS) /= RHS;
}

// CHECK-LABEL: @signed_char_div_signed_char_unsigned_int
void signed_char_div_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = udiv i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3100
  (*LHS) /= RHS;
}

// CHECK-LABEL: @signed_char_div_signed_char_signed_int
void signed_char_div_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = sdiv i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3200
  (*LHS) /= RHS;
}

//----------------------------------------------------------------------------//
// Compound remainder operator.                                               //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_rem_signed_char_unsigned_char
void unsigned_char_rem_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = srem i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3300
  (*LHS) %= RHS;
}

// CHECK-LABEL: @unsigned_char_rem_signed_char_signed_char
void unsigned_char_rem_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = srem i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3400
  (*LHS) %= RHS;
}

// CHECK-LABEL: @unsigned_char_rem_signed_char_unsigned_int
void unsigned_char_rem_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = urem i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3500_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3500_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3500
  (*LHS) %= RHS;
}

// CHECK-LABEL: @unsigned_char_rem_signed_char_signed_int
void unsigned_char_rem_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = srem i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3600
  (*LHS) %= RHS;
}

// CHECK-LABEL: @signed_char_rem_unsigned_char
void signed_char_rem_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = srem i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3700
  (*LHS) %= RHS;
}

// CHECK-LABEL: @signed_char_rem_signed_char
void signed_char_rem_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = srem i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3800
  (*LHS) %= RHS;
}

// CHECK-LABEL: @signed_char_rem_signed_char_unsigned_int
void signed_char_rem_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = urem i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_3900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_3900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 3900
  (*LHS) %= RHS;
}

// CHECK-LABEL: @signed_char_rem_signed_char_signed_int
void signed_char_rem_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = srem i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4000
  (*LHS) %= RHS;
}

//----------------------------------------------------------------------------//
// Compound left-shift operator.                                              //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_shl_signed_char_unsigned_char
void unsigned_char_shl_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4100
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @unsigned_char_shl_signed_char_signed_char
void unsigned_char_shl_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4200
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @unsigned_char_shl_signed_char_unsigned_int
void unsigned_char_shl_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4300
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @unsigned_char_shl_signed_char_signed_int
void unsigned_char_shl_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4400
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @signed_char_shl_unsigned_char
void signed_char_shl_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4500
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @signed_char_shl_signed_char
void signed_char_shl_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4600
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @signed_char_shl_signed_char_unsigned_int
void signed_char_shl_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4700
  (*LHS) <<= RHS;
}

// CHECK-LABEL: @signed_char_shl_signed_char_signed_int
void signed_char_shl_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = shl i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4800
  (*LHS) <<= RHS;
}

//----------------------------------------------------------------------------//
// Compound right-shift operator.                                             //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_shr_signed_char_unsigned_char
void unsigned_char_shr_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_4900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_4900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 4900
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @unsigned_char_shr_signed_char_signed_char
void unsigned_char_shr_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5000
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @unsigned_char_shr_signed_char_unsigned_int
void unsigned_char_shr_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5100
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @unsigned_char_shr_signed_char_signed_int
void unsigned_char_shr_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5200
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @signed_char_shr_unsigned_char
void signed_char_shr_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5300
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @signed_char_shr_signed_char
void signed_char_shr_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5400
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @signed_char_shr_signed_char_unsigned_int
void signed_char_shr_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5500
  (*LHS) >>= RHS;
}

// CHECK-LABEL: @signed_char_shr_signed_char_signed_int
void signed_char_shr_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = ashr i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5600
  (*LHS) >>= RHS;
}

//----------------------------------------------------------------------------//
// Compound and operator.                                                     //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_and_signed_char_unsigned_char
void unsigned_char_and_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5700
  (*LHS) &= RHS;
}

// CHECK-LABEL: @unsigned_char_and_signed_char_signed_char
void unsigned_char_and_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5800
  (*LHS) &= RHS;
}

// CHECK-LABEL: @unsigned_char_and_signed_char_unsigned_int
void unsigned_char_and_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_5900_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_5900_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 5900
  (*LHS) &= RHS;
}

// CHECK-LABEL: @unsigned_char_and_signed_char_signed_int
void unsigned_char_and_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6000
  (*LHS) &= RHS;
}

// CHECK-LABEL: @signed_char_and_unsigned_char
void signed_char_and_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6100
  (*LHS) &= RHS;
}

// CHECK-LABEL: @signed_char_and_signed_char
void signed_char_and_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6200
  (*LHS) &= RHS;
}

// CHECK-LABEL: @signed_char_and_signed_char_unsigned_int
void signed_char_and_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6300
  (*LHS) &= RHS;
}

// CHECK-LABEL: @signed_char_and_signed_char_signed_int
void signed_char_and_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = and i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6400
  (*LHS) &= RHS;
}

//----------------------------------------------------------------------------//
// Compound xor operator.                                                     //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_or_signed_char_unsigned_char
void unsigned_char_or_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6500_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6500
  (*LHS) |= RHS;
}

// CHECK-LABEL: @unsigned_char_or_signed_char_signed_char
void unsigned_char_or_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6600
  (*LHS) |= RHS;
}

// CHECK-LABEL: @unsigned_char_or_signed_char_unsigned_int
void unsigned_char_or_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6700_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6700_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6700
  (*LHS) |= RHS;
}

// CHECK-LABEL: @unsigned_char_or_signed_char_signed_int
void unsigned_char_or_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6800
  (*LHS) |= RHS;
}

// CHECK-LABEL: @signed_char_or_unsigned_char
void signed_char_or_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_6900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_6900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 6900
  (*LHS) |= RHS;
}

// CHECK-LABEL: @signed_char_or_signed_char
void signed_char_or_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7000
  (*LHS) |= RHS;
}

// CHECK-LABEL: @signed_char_or_signed_char_unsigned_int
void signed_char_or_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7100_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7100
  (*LHS) |= RHS;
}

// CHECK-LABEL: @signed_char_or_signed_char_signed_int
void signed_char_or_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = or i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7200_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7200
  (*LHS) |= RHS;
}

//----------------------------------------------------------------------------//
// Compound or operator.                                                      //
//----------------------------------------------------------------------------//

// CHECK-LABEL: @unsigned_char_xor_signed_char_unsigned_char
void unsigned_char_xor_signed_char_unsigned_char(unsigned char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7300_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7300
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @unsigned_char_xor_signed_char_signed_char
void unsigned_char_xor_signed_char_signed_char(unsigned char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7400_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7400
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @unsigned_char_xor_signed_char_unsigned_int
void unsigned_char_xor_signed_char_unsigned_int(unsigned char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7500_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7500_UNSIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7500
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @unsigned_char_xor_signed_char_signed_int
void unsigned_char_xor_signed_char_signed_int(unsigned char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = zext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = zext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7600_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7600
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @signed_char_xor_unsigned_char
void signed_char_xor_unsigned_char(signed char *LHS, unsigned char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = zext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7700_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7700
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @signed_char_xor_signed_char
void signed_char_xor_signed_char(signed char *LHS, signed char RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i8, align 1
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i8 %[[ARG1:.*]], ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHS:.*]] = load i8, ptr %[[RHSADDR]], align 1
  // CHECK-NEXT: %[[RHSEXT:.*]] = sext i8 %[[RHS]] to i32
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHSEXT]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7800_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7800
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @signed_char_xor_signed_char_unsigned_int
void signed_char_xor_signed_char_unsigned_int(signed char *LHS, unsigned int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_7900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_7900_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 7900
  (*LHS) ^= RHS;
}

// CHECK-LABEL: @signed_char_xor_signed_char_signed_int
void signed_char_xor_signed_char_signed_int(signed char *LHS, signed int RHS) {
  // CHECK: {
  // CHECK-NEXT: entry:
  // CHECK-NEXT: %[[ADDRESS:.*]] = alloca ptr, align 8
  // CHECK-NEXT: %[[RHSADDR:.*]] = alloca i32, align 4
  // CHECK-NEXT: store ptr %[[ARG0:.*]], ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: store i32 %[[ARG1:.*]], ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[RHS:.*]] = load i32, ptr %[[RHSADDR]], align 4
  // CHECK-NEXT: %[[LHSADDR:.*]] = load ptr, ptr %[[ADDRESS]], align 8
  // CHECK-NEXT: %[[LHS:.*]] = load i8, ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: %[[LHSEXT:.*]] = sext i8 %[[LHS]] to i32
  // CHECK-NEXT: %[[SRC:.*]] = xor i32 %[[LHSEXT]], %[[RHS]]
  // CHECK-NEXT: %[[DST:.*]] = trunc i32 %[[SRC]] to i8
  // CHECK-SANITIZE-NEXT: %[[ANYEXT:.*]] = sext i8 %[[DST]] to i32, !nosanitize
  // CHECK-SANITIZE-NEXT: %[[TRUNCHECK:.*]] = icmp eq i32 %[[ANYEXT]], %[[SRC]], !nosanitize
  // CHECK-SANITIZE-NEXT: br i1 %[[TRUNCHECK]], label %[[CONT:.*]], label %[[HANDLER_IMPLICIT_CONVERSION:[^,]+]],{{.*}} !nosanitize
  // CHECK-SANITIZE: [[HANDLER_IMPLICIT_CONVERSION]]:
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTSRC:.*]] = zext i32 %[[SRC]] to i64, !nosanitize
  // CHECK-SANITIZE-ANYRECOVER-NEXT: %[[EXTDST:.*]] = zext i8 %[[DST]] to i64, !nosanitize
  // CHECK-SANITIZE-NORECOVER-NEXT: call void @__ubsan_handle_implicit_conversion_abort(ptr @[[LINE_8000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-RECOVER-NEXT: call void @__ubsan_handle_implicit_conversion(ptr @[[LINE_8000_SIGNED_TRUNCATION]], i64 %[[EXTSRC]], i64 %[[EXTDST]]){{.*}}, !nosanitize
  // CHECK-SANITIZE-TRAP-NEXT: call void @llvm.ubsantrap(i8 7){{.*}}, !nosanitize
  // CHECK-SANITIZE-UNREACHABLE-NEXT: unreachable, !nosanitize
  // CHECK-SANITIZE: [[CONT]]:
  // CHECK-NEXT: store i8 %[[DST]], ptr %[[LHSADDR]], align 1
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
#line 8000
  (*LHS) ^= RHS;
}
