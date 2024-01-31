// Check that -fsanitize=signed-integer-wrap instruments with -fwrapv
// RUN: %clang_cc1 -fwrapv -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=signed-integer-wrap | FileCheck %s --check-prefix=CHECKSIW

// Check that -fsanitize=signed-integer-overflow doesn't instrument with -fwrapv
// RUN: %clang_cc1 -fwrapv -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=signed-integer-overflow | FileCheck %s --check-prefix=CHECKSIO

extern volatile int a, b, c;

// CHECKSIW-LABEL: define void @test_add_overflow
void test_add_overflow(void) {
  // CHECKSIW: [[ADD0:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[ADD1:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[ADD2:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[ADD0]], i32 [[ADD1]])
  // CHECKSIW: [[ADD4:%.*]] = extractvalue { i32, i1 } [[ADD2]], 1
  // CHECKSIW-NEXT: [[ADD5:%.*]] = xor i1 [[ADD4]], true
  // CHECKSIW-NEXT: br i1 [[ADD5]], {{.*}} %handler.add_overflow
  // CHECKSIW: call void @__ubsan_handle_add_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_add_overflow
  a = b + c;
}

// CHECKSIW-LABEL: define void @test_inc_overflow
void test_inc_overflow(void) {
  // This decays and gets handled by __ubsan_handle_add_overflow...
  // CHECKSIW: [[INC0:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[INC1:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 [[INC0]], i32 1)
  // CHECKSIW: [[INC3:%.*]] = extractvalue { i32, i1 } [[INC1]], 1
  // CHECKSIW-NEXT: [[INC4:%.*]] = xor i1 [[INC3]], true
  // CHECKSIW-NEXT: br i1 [[INC4]], {{.*}} %handler.add_overflow
  // CHECKSIW: call void @__ubsan_handle_add_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_add_overflow
  a++;
}

// CHECKSIW-LABEL: define void @test_sub_overflow
void test_sub_overflow(void) {
  // CHECKSIW: [[SUB0:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[SUB1:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[SUB2:%.*]] = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 [[SUB0]], i32 [[SUB1]])
  // CHECKSIW: [[SUB4:%.*]] = extractvalue { i32, i1 } [[SUB2]], 1
  // CHECKSIW-NEXT: [[SUB5:%.*]] = xor i1 [[SUB4]], true
  // CHECK-NEXT br i1 [[SUB5]], {{.*}} %handler.sub_overflow
  // CHECKSIW: call void @__ubsan_handle_sub_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_sub_overflow
  a = b - c;
}

// CHECKSIW-LABEL: define void @test_mul_overflow
void test_mul_overflow(void) {
  // CHECKSIW: [[MUL0:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[MUL1:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[MUL2:%.*]] = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 [[MUL0]], i32 [[MUL1]])
  // CHECKSIW: [[MUL4:%.*]] = extractvalue { i32, i1 } [[MUL2]], 1
  // CHECK-NEXT [[MUL5:%.*]] = xor i1 [[MUL4]], true
  // CHECK-NEXT br i1 [[MUL5]], {{.*}} %handler.mul_overflow
  // CHECKSIW: call void @__ubsan_handle_mul_overflow

  // CHECKSIO-NOT: call void @__ubsan_handle_mul_overflow
  a = b * c;
}

// CHECKSIW-LABEL: define void @test_div_overflow
void test_div_overflow(void) {
  // CHECKSIW: [[DIV0:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[DIV1:%.*]] = load {{.*}} i32
  // CHECKSIW-NEXT: [[DIV2:%.*]] = icmp ne i32 [[DIV0]], -2147483648
  // CHECKSIW-NEXT: [[DIV3:%.*]] = icmp ne i32 [[DIV1]], -1
  // CHECKSIW-NEXT: [[DIVOR:%or]] = or i1 [[DIV2]], [[DIV3]]
  // CHECKSIW-NEXT: br {{.*}} %handler.divrem_overflow

  // -fsanitize=signed-integer-overflow still instruments division even with -fwrapv
  // CHECKSIO: br {{.*}} %handler.divrem_overflow
  a = b / c;
}
