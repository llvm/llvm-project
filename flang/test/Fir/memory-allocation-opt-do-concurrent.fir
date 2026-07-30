// The default (-fno-stack-arrays) policy sends every runtime-sized array to the
// heap, reproducing the legacy memory-allocation-opt "dynamic-array-on-heap"
// behavior inside do_concurrent loops (with conditional allocations).
// RUN: fir-opt --allocation-placement %s | FileCheck %s

func.func @test_do_concurrent() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_concurrent {
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) {
      %0 = fir.alloca !fir.array<?xf32>, %arg0
      fir.call @bar(%0) : (!fir.ref<!fir.array<?xf32>>) -> ()
    }
  }
  return
}
// CHECK-LABEL: func.func @test_do_concurrent() {
// CHECK: fir.do_concurrent {
// CHECK:   fir.do_concurrent.loop
// CHECK:     %[[MEM:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "", uniq_name = ""}
// CHECK:     %[[REF:.*]] = fir.convert %[[MEM]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:     fir.call @bar(%[[REF]])
// CHECK:     fir.freemem %[[MEM]] : !fir.heap<!fir.array<?xf32>>
// CHECK:   }
// CHECK: }

func.func @test_do_concurrent_if(%cond: i1) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_concurrent {
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) {
      fir.if %cond {
        %0 = fir.alloca !fir.array<?xf32>, %arg0
        fir.call @bar(%0) : (!fir.ref<!fir.array<?xf32>>) -> ()
      }
    }
  }
  return
}
// CHECK-LABEL: func.func @test_do_concurrent_if(
// CHECK: fir.do_concurrent {
// CHECK:   fir.do_concurrent.loop
// CHECK:     %[[PTR:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>>
// CHECK:     %[[NULL:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:     fir.store %[[NULL]] to %[[PTR]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:     fir.if
// CHECK:       %[[PREV:.*]] = fir.load %[[PTR]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:       fir.if {{.*}} {
// CHECK:         fir.freemem %[[PREV]] : !fir.heap<!fir.array<?xf32>>
// CHECK:       }
// CHECK:       %[[MEM:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "", uniq_name = ""}
// CHECK:       %[[REF:.*]] = fir.convert %[[MEM]] : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:       fir.store %[[MEM]] to %[[PTR]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:       fir.call @bar(%[[REF]])
// CHECK:     %[[LAST:.*]] = fir.load %[[PTR]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:     fir.if {{.*}} {
// CHECK:       fir.freemem %[[LAST]] : !fir.heap<!fir.array<?xf32>>
// CHECK:     }
// CHECK:   }
// CHECK: }

func.func @test_do_concurrent_if_else(%cond: i1) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_concurrent {
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) {
      fir.if %cond {
        %0 = fir.alloca !fir.array<?xf32>, %arg0
        fir.call @bar(%0) : (!fir.ref<!fir.array<?xf32>>) -> ()
      } else {
        %1 = fir.alloca !fir.array<?xf32>, %arg0
        fir.call @bar(%1) : (!fir.ref<!fir.array<?xf32>>) -> ()
      }
    }
  }
  return
}
// CHECK-LABEL: func.func @test_do_concurrent_if_else(
// CHECK: fir.do_concurrent {
// CHECK:   fir.do_concurrent.loop
// CHECK:     %[[PTR_ELSE:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>>
// CHECK:     %[[NULL_ELSE:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:     fir.store %[[NULL_ELSE]] to %[[PTR_ELSE]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:     %[[PTR_THEN:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>>
// CHECK:     %[[NULL_THEN:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:     fir.store %[[NULL_THEN]] to %[[PTR_THEN]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:     fir.if
// CHECK:       %[[PREV_THEN:.*]] = fir.load %[[PTR_THEN]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:       fir.if {{.*}} {
// CHECK:         fir.freemem %[[PREV_THEN]] : !fir.heap<!fir.array<?xf32>>
// CHECK:       }
// CHECK:       %[[MEM_THEN:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "", uniq_name = ""}
// CHECK:       fir.store %[[MEM_THEN]] to %[[PTR_THEN]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:       fir.call @bar(
// CHECK:     } else {
// CHECK:       %[[PREV_ELSE:.*]] = fir.load %[[PTR_ELSE]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:       fir.if {{.*}} {
// CHECK:         fir.freemem %[[PREV_ELSE]] : !fir.heap<!fir.array<?xf32>>
// CHECK:       }
// CHECK:       %[[MEM_ELSE:.*]] = fir.allocmem !fir.array<?xf32>, %{{.*}} {bindc_name = "", uniq_name = ""}
// CHECK:       fir.store %[[MEM_ELSE]] to %[[PTR_ELSE]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:       fir.call @bar(
// CHECK:     %[[LAST_THEN:.*]] = fir.load %[[PTR_THEN]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:     fir.if {{.*}} {
// CHECK:       fir.freemem %[[LAST_THEN]] : !fir.heap<!fir.array<?xf32>>
// CHECK:     }
// CHECK:     %[[LAST_ELSE:.*]] = fir.load %[[PTR_ELSE]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
// CHECK:     fir.if {{.*}} {
// CHECK:       fir.freemem %[[LAST_ELSE]] : !fir.heap<!fir.array<?xf32>>
// CHECK:     }
// CHECK:   }
// CHECK: }

func.func private @bar(!fir.ref<!fir.array<?xf32>>)
