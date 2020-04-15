// Test lowering FIR to LLVM IR of fir.select{|_rank|_case}

// RUN: tco %s | FileCheck %s

// CHECK: @x({{.*}} %[[ADDR:.*]])
func @x(%addr : !fir.ref<!fir.array<10x10xi32>>) -> index {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index
  // CHECK: %[[ROW:.*]] = phi i64
  // CHECK: icmp slt i64 %[[ROW]], 11
  fir.do_loop %iv = %c0 to %c10 step %c1 {
    // CHECK: %[[COL:.*]] = phi i64
    // CHECK: icmp slt i64 %[[COL]], 11
    fir.do_loop %jv = %c0 to %c10 step %c1 {
      // CHECK: getelementptr {{.*}} %[[ADDR]], i64 0, i64 %[[ROW]], i64 %[[COL]]
      %ptr = fir.coordinate_of %addr, %jv, %iv : (!fir.ref<!fir.array<10x10xi32>>, index, index) -> !fir.ref<i32>
      %c22 = constant 22 : i32
      // CHECK: store i32 22,
      fir.store %c22 to %ptr : !fir.ref<i32>
    }
  }
  // CHECK: ret i64 10
  return %c10 : index
}
