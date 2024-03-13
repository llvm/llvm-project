// RUN: %clang_cc1 -O1 %s -o - -emit-llvm -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -mllvm -ubsan-exp-hot | FileCheck %s
// RUN: %clang_cc1 -O1 %s -o - -emit-llvm -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -mllvm -ubsan-exp-hot -mllvm -clang-remove-traps -mllvm -remove-traps-random-rate=1 %s -o - | FileCheck %s --check-prefixes=REMOVE

#include <stdbool.h>

int test(int x) {
  return x + 123;
}

// CHECK-LABEL: define {{.*}}i32 @test(
// CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(
// CHECK: trap:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 0)
// CHECK-NEXT: unreachable

// REMOVE-LABEL: define {{.*}}i32 @test(
// REMOVE: add i32 %x, 123
// REMOVE-NEXT: ret i32


bool experimental_hot() __asm("llvm.experimental.hot");

bool test_asm() {
  return experimental_hot();
}

// CHECK-LABEL: define {{.*}}i1 @test_asm(
// CHECK: [[R:%.*]] = tail call zeroext i1 @llvm.experimental.hot()
// CHECK: ret i1 [[R]]

// REMOVE-LABEL: define {{.*}}i1 @test_asm(
// REMOVE: ret i1 true
