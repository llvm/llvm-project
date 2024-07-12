// RUN: fir-opt %s | tco | FileCheck %s

// CHECK-LABEL: define i32 @f1(i32 %0, i32 %1)
func.func @f1(%a : i32, %b : i32) -> i32 {

  // CHECK: %[[reg3:.*]] = add i32 %0, %1
  %1 = arith.addi %a, %b : i32
  %2 = arith.addi %b, %a : i32
  // CHECK:  mul i32 %[[reg3]], %[[reg3]]
  %3 = arith.muli %1, %2 : i32
  return %3 : i32
}

// CHECK-LABEL: define i32 @f2(ptr %0)
func.func @f2(%a : !fir.ref<i32>) -> i32 {
  %1 = fir.load %a : !fir.ref<i32>
  // CHECK: %[[r2:.*]] = load
  %2 = fir.load %a : !fir.ref<i32>
  // CHECK: %[[r3:.*]] = add i32 %[[r2]], %[[r2]]
  %3 = arith.addi %1, %2 : i32
  %4 = fir.load %a : !fir.ref<i32>
  // CHECK: %[[r4:.*]] = add i32 %[[r3]], %[[r2]]
  %5 = arith.addi %3, %4 : i32
  %6 = fir.load %a : !fir.ref<i32>
  // CHECK: %[[r5:.*]] = add i32 %[[r4]], %[[r2]]
  %7 = arith.addi %5, %6 : i32
  %8 = fir.load %a : !fir.ref<i32>
  // CHECK: %[[r6:.*]] = add i32 %[[r5]], %[[r2]]
  %9 = arith.addi %7, %8 : i32
  %10 = fir.load %a : !fir.ref<i32>
  // CHECK: %[[r7:.*]] = add i32 %[[r2]], %[[r6]]
  %11 = arith.addi %10, %9 : i32
  return %11 : i32
}
