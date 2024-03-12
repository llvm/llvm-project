// RUN: %clang_cc1 -O1 -emit-llvm -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow %s -o - | FileCheck %s 
// RUN: %clang_cc1 -O1 -emit-llvm -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -mllvm -clang-remove-traps -mllvm -remove-traps-random-rate=1 %s -o - | FileCheck %s --implicit-check-not="call void @llvm.ubsantrap" --check-prefixes=REMOVE

int test(int x) {
  return x + 123;
}

// CHECK-LABEL: define {{.*}}i32 @test(
// CHECK: call { i32, i1 } @llvm.sadd.with.overflow.i32(
// CHECK: trap:
// CHECK-NEXT: call void @llvm.ubsantrap(i8 0)
// CHECK-NEXT: unreachable

// REMOVE-LABEL: define {{.*}}i32 @test(
// REMOVE: call { i32, i1 } @llvm.sadd.with.overflow.i32(
