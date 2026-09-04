// RUN: inter-opt %s --inter-prepare-counted-loops | FileCheck %s

func.func @counted(%lb: i64, %ub: i64, %step: i64, %init: i32) -> i32 {
  %result:2 = scf.while (%iv = %lb, %value = %init) : (i64, i32) -> (i64, i32) {
    %condition = llvm.icmp "slt" %iv, %ub : i64
    scf.condition(%condition) %iv, %value : i64, i32
  } do {
  ^bb0(%iv: i64, %value: i32):
    %next = llvm.add %iv, %step : i64
    %updated = arith.addi %value, %value : i32
    scf.yield %next, %updated : i64, i32
  }
  return %result#1 : i32
}

// CHECK-LABEL: func.func @counted
// CHECK: %[[RESULT:.*]] = scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[VALUE:.*]] = %{{.*}}) -> (i32) : i64 {
// CHECK:   %[[UPDATED:.*]] = arith.addi %[[VALUE]], %[[VALUE]] : i32
// CHECK:   scf.yield %[[UPDATED]] : i32
// CHECK: return %[[RESULT]] : i32
