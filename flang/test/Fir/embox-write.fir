// RUN: fir-opt %s | tco | FileCheck %s

// CHECK-LABEL: @set_all_n
func.func @set_all_n(%n : index, %x : i32) {
  %aTmp = fir.alloca i32, %n
  %aMem = fir.convert %aTmp : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
  %c1 = arith.constant 1 : index
  %aDim = fir.shape %n : (index) -> !fir.shape<1>
  %a = fir.embox %aMem(%aDim) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  // CHECK-DAG: %[[IV:.*]] = phi i64
  // CHECK-DAG: %[[LCV:.*]] = phi i64
  // CHECK: icmp sgt i64 %[[LCV]], 0
  fir.do_loop %i = %c1 to %n step %c1 unordered {
    %1 = fir.coordinate_of %a, %i : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    // CHECK: store i32 %{{.*}}, ptr %{{.*}}
    fir.store %x to %1 : !fir.ref<i32>
  }
  return
}
