// Test lowering FIR to LLVM IR of fir.select{|_rank|_case}

// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @x
func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  // CHECK: [[LOOP:[0-9]+]]:
  // CHECK: %[[COND:.*]] = icmp slt i64
  // CHECK: br i1 %[[COND]]
  fir.do_loop %iv = %lb to %ub step %step unordered {
    // expect following conditional blocks to get fused
    // CHECK: select i1 %
    fir.if %b {
      // CHECK: store i64
      fir.store %iv to %addr : !fir.ref<index>
    } else {
      %zero = constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
    // CHECK: br label %[[LOOP]]
  }
  // CHECK: ret void
  return
}
