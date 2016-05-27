// RUN: %clang_cc1 -triple x86_64-linux-gnu -fms-extensions %s -emit-llvm -o - | FileCheck %s

long test1(long *p) {
  return _InterlockedIncrement(p);
}
// CHECK-DAG: define i64 @test1(
// CHECK:   %[[p_addr:.*]] = alloca i64*, align 8
// CHECK:   store i64* %p, i64** %[[p_addr]], align 8
// CHECK:   %[[p_load:.*]] = load i64*, i64** %[[p_addr]], align 8
// CHECK:   %[[atomic_add:.*]] = atomicrmw volatile add i64* %[[p_load]], i64 1 seq_cst
// CHECK:   %[[res:.*]] = add i64 %[[atomic_add]], 1
// CHECK:   ret i64 %[[res]]

long test2(long *p) {
  return _InterlockedDecrement(p);
}
// CHECK-DAG: define i64 @test2(
// CHECK:   %[[p_addr:.*]] = alloca i64*, align 8
// CHECK:   store i64* %p, i64** %[[p_addr]], align 8
// CHECK:   %[[p_load:.*]] = load i64*, i64** %[[p_addr]], align 8
// CHECK:   %[[atomic_sub:.*]] = atomicrmw volatile sub i64* %[[p_load]], i64 1 seq_cst
// CHECK:   %[[res:.*]] = sub i64 %[[atomic_sub]], 1
// CHECK:   ret i64 %[[res]]
