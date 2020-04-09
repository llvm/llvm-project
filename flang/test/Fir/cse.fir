// Test CSE pass

// RUN: tco -emit-fir %s | tco | FileCheck %s

// CHECK-LABEL: @fun
func @fun(%a : !fir.ref<i64>) -> i64 {
  // CHECK: load i64
  %1 = fir.load %a : !fir.ref<i64>
  %2 = fir.load %a : !fir.ref<i64>
  // CHECK-NOT: load i64
  // CHECK-COUNT-6: add i64
  %3 = addi %1, %2 : i64
  %4 = fir.load %a : !fir.ref<i64>
  %5 = addi %3, %4 : i64
  %6 = fir.load %a : !fir.ref<i64>
  %7 = addi %5, %6 : i64
  %8 = fir.load %a : !fir.ref<i64>
  %9 = addi %7, %8 : i64
  %10 = fir.load %a : !fir.ref<i64>
  %11 = addi %10, %9 : i64
  %12 = fir.load %a : !fir.ref<i64>
  %13 = addi %11, %12 : i64
  // CHECK-NEXT: ret i64
  return %13 : i64
}

// CHECK-LABEL: @bar
func @bar(%a : !fir.ref<i64>) -> i64

// CHECK-LABEL: @fun2
func @fun2(%a : !fir.ref<i64>) -> i64 {
  // CHECK: load i64
  %1 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: call i64
  %2 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  // CHECK-COUNT-6: add i64
  %3 = addi %1, %2 : i64
  %4 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %5 = addi %3, %4 : i64
  %6 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %7 = addi %5, %6 : i64
  %8 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %9 = addi %7, %8 : i64
  %10 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %11 = addi %10, %9 : i64
  %12 = fir.call @bar(%a) { pure = true } : (!fir.ref<i64>) -> i64
  %13 = addi %11, %12 : i64
  // CHECK-NEXT: ret i64
  return %13 : i64
}
