// RUN: tco %s | FileCheck %s

// CHECK: @x({{.*}} %[[ADDR:.*]])
func.func @x(%addr : !fir.ref<!fir.array<10x10xi32>>) -> index {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[R:.*]] = phi i64 {{.*}} [ 0,
  // CHECK-DAG: %[[ROW:.*]] = phi i64 {{.*}} [ 11,
  // CHECK: icmp sgt i64 %[[ROW]], 0
  fir.do_loop %iv = %c0 to %c10 step %c1 {
    // CHECK-DAG: %[[C:.*]] = phi i64 {{.*}} [ 0,
    // CHECK-DAG: %[[COL:.*]] = phi i64 {{.*}} [ 11,
    // CHECK: icmp sgt i64 %[[COL]], 0
    fir.do_loop %jv = %c0 to %c10 step %c1 {
      // CHECK: getelementptr {{.*}} %[[ADDR]], i32 0, i64 %[[R]], i64 %[[C]]
      %ptr = fir.coordinate_of %addr, %jv, %iv : (!fir.ref<!fir.array<10x10xi32>>, index, index) -> !fir.ref<i32>
      %c22 = arith.constant 22 : i32
      // CHECK: store i32 22,
      fir.store %c22 to %ptr : !fir.ref<i32>
    }
  }
  // CHECK: ret i64 10
  return %c10 : index
}
