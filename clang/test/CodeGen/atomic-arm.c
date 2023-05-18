// RUN: %clang_cc1 -triple thumbv6m-apple-unknown-macho %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-V6M
// RUN: %clang_cc1 -triple thumbv7m-apple-unknown-macho %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-V7M
// RUN: %clang_cc1 -triple thumbv7-apple-ios13.0 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-HOSTED
// RUN: %clang_cc1 -triple thumbv7k-apple-watchos5.0 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,CHECK-HOSTED


// CHECK-V6M: @always1 = global i32 0
// CHECK-V6M: @always4 = global i32 0
// CHECK-V6M: @always8 = global i32 0

// CHECK-V7M: @always1 = global i32 1
// CHECK-V7M: @always4 = global i32 1
// CHECK-V7M: @always8 = global i32 0

// CHECK-HOSTED: @always1 = global i32 1
// CHECK-HOSTED: @always4 = global i32 1
// CHECK-HOSTED: @always8 = global i32 1

int always1 = __atomic_always_lock_free(1, 0);
int always4 = __atomic_always_lock_free(4, 0);
int always8 = __atomic_always_lock_free(8, 0);

int lock_free_1() {
  // CHECK-LABEL: @lock_free_1
  // CHECK-V6M:   [[RES:%.*]] = call arm_aapcscc zeroext i1 @__atomic_is_lock_free(i32 noundef 1, ptr noundef null)
  // CHECK-V6M:   [[RES32:%.*]] = zext i1 [[RES]] to i32
  // CHECK-V6M:   ret i32 [[RES32]]

  // CHECK-V7M: ret i32 1
  // CHECK-HOSTED: ret i32 1
  return __c11_atomic_is_lock_free(1);
}

int lock_free_4() {
  // CHECK-LABEL: @lock_free_4
  // CHECK-V6M:   [[RES:%.*]] = call arm_aapcscc zeroext i1 @__atomic_is_lock_free(i32 noundef 4, ptr noundef null)
  // CHECK-V6M:   [[RES32:%.*]] = zext i1 [[RES]] to i32
  // CHECK-V6M:   ret i32 [[RES32]]

  // CHECK-V7M: ret i32 1
  // CHECK-HOSTED: ret i32 1
  return __c11_atomic_is_lock_free(4);
}

int lock_free_8() {
  // CHECK-LABEL: @lock_free_8
  // CHECK-V6M:   [[RES:%.*]] = call arm_aapcscc zeroext i1 @__atomic_is_lock_free(i32 noundef 8, ptr noundef null)
  // CHECK-V6M:   [[RES32:%.*]] = zext i1 [[RES]] to i32
  // CHECK-V6M:   ret i32 [[RES32]]

  // CHECK-V7M:   [[RES:%.*]] = call arm_aapcscc zeroext i1 @__atomic_is_lock_free(i32 noundef 8, ptr noundef null)
  // CHECK-V7M:   [[RES32:%.*]] = zext i1 [[RES]] to i32
  // CHECK-V7M:   ret i32 [[RES32]]

  // CHECK-HOSTED: ret i32 1
  return __c11_atomic_is_lock_free(8);
}
