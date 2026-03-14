// Test when the saved mask is used in the same run to save
// another value (like the RHS).
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @saving_mask_and_rhs(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  hlfir.where {
    %2 = hlfir.elemental %0 : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
    ^bb0(%arg1: index):
      %3 = hlfir.designate %1#0 (%arg1)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %4 = fir.load %3 : !fir.ref<i32>
      %5 = arith.cmpi sgt, %4, %c0_i32 : i32
      %6 = fir.convert %5 : (i1) -> !fir.logical<4>
      hlfir.yield_element %6 : !fir.logical<4>
    }
    hlfir.yield %2 : !hlfir.expr<10x!fir.logical<4>> cleanup {
      hlfir.destroy %2 : !hlfir.expr<10x!fir.logical<4>>
    }
  } do {
    hlfir.region_assign {
      hlfir.yield %1#0 : !fir.ref<!fir.array<10xi32>>
    } to {
      %2 = hlfir.designate %1#0 (%c10:%c1:%c-1)  shape %0 : (!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
      hlfir.yield %2 : !fir.box<!fir.array<10xi32>>
    }
  }
  return
}


// Creating mask temporary.

// CHECK-LABEL:   func.func @saving_mask_and_rhs(
// CHECK:  %[[VAL_8:.*]] = hlfir.elemental {{.*}} !hlfir.expr<10x!fir.logical<4>>
// CHECK:  %[[VAL_14:.*]]:3 = hlfir.associate %[[VAL_8]]({{.*}}) {uniq_name = ".tmp.where"}

// Creating RHS temporary using the mask temporary (and not the hlfir.elemental)

// CHECK:  %[[VAL_25:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp.where", uniq_name = ""}
// CHECK:  %[[VAL_27:.*]]:2 = hlfir.declare %[[VAL_25]]({{.*}}) {uniq_name = ".tmp.where"}
// CHECK:  fir.do_loop
// CHECK:    %[[VAL_29:.*]] = hlfir.designate %[[VAL_14]]#0 ({{.*}})
// CHECK:    %[[VAL_30:.*]] = fir.load %[[VAL_29]] : !fir.ref<!fir.logical<4>>
// CHECK:    %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (!fir.logical<4>) -> i1
// CHECK:    fir.if %[[VAL_31]] {
// CHECK:      %[[VAL_36:.*]] = hlfir.designate %[[VAL_27]]#0 ({{.*}})
// CHECK:      hlfir.assign %{{.*}} to %[[VAL_36]] : i32, !fir.ref<i32>
// CHECK:    }
// CHECK:  }

func.func @forall_mask_and_rhs(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  }  (%arg1: index) {
    hlfir.forall_mask {
      %2 = arith.subi %c11, %arg1 : index
      %3 = hlfir.designate %1#0 (%2)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %4 = fir.load %3 : !fir.ref<i32>
      %5 = arith.cmpi sgt, %4, %c0_i32 : i32
      hlfir.yield %5 : i1
    } do {
      hlfir.region_assign {
        %2 = arith.subi %c11, %arg1 : index
        %3 = hlfir.designate %1#0 (%2)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
        %4 = fir.load %3 : !fir.ref<i32>
        hlfir.yield %4 : i32
      } to {
        %2 = hlfir.designate %1#0 (%arg1)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
        hlfir.yield %2 : !fir.ref<i32>
      }
    }
  }
  return
}

// The mask and rhs are saved in the same loop, the mask value is a scalar i1
// and it can be used directly to mask the rhs (instead of loading the temp).

// CHECK-LABEL:   func.func @forall_mask_and_rhs(
// CHECK:           %[[VAL_18:.*]] = fir.allocmem !fir.array<?xi1>, %{{.*}} {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_18]](%{{.*}}) {uniq_name = ".tmp.forall"}
// CHECK:           %[[VAL_29:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[VAL_31:.*]]:2 = hlfir.declare %[[VAL_29]](%{{.*}}) {uniq_name = ".tmp.forall"}
// CHECK:             %[[VAL_36:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
// CHECK:             %[[VAL_39:.*]] = hlfir.designate %[[VAL_20]]#0 (%{{.*}})
// CHECK:             hlfir.assign %[[VAL_36]] to %[[VAL_39]] : i1, !fir.ref<i1>
// CHECK:             fir.if %[[VAL_36]] {
// CHECK:               %[[VAL_45:.*]] = hlfir.designate %[[VAL_31]]#0 (%{{.*}})
// CHECK:               hlfir.assign %{{.*}} to %[[VAL_45]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
