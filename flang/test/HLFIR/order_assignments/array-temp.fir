// Test that hlfir.where/hlfir.forall temporary storages used for simple
// scalar values are lowered to a multidimensional ArrayTemp directly indexed
// by the enclosing loop induction variables (using hlfir.designate on a
// fir.shape_shift) rather than to a 1D HomogeneousScalarStack with a counter.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

// Single-dimension case: the saved RHS is stored in a 1D ArrayTemp indexed
// by the where loop induction variable, with a fir.shape_shift instead of a
// fir.shape.
func.func @where_self_overlap(%x: !fir.ref<!fir.array<10xi32>>, %mask: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %2 = hlfir.designate %x (%c10:%c1:%c-1)  shape %1 :
(!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
      hlfir.yield %2 : !fir.ref<!fir.array<10xi32>>
    } to {
      hlfir.yield %x : !fir.ref<!fir.array<10xi32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @where_self_overlap(
// CHECK-NOT:       fir.alloca index
// CHECK:           %[[ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp.where", uniq_name = ""}
// CHECK:           %[[SHAPE:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = ".tmp.where"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// First loop: save the RHS values, addressed by the loop induction variable.
// CHECK:           fir.do_loop %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             fir.if %{{.*}} {
// CHECK-NOT:           fir.load %{{.*}} : !fir.ref<index>
// CHECK:               %[[ADDR0:.*]] = hlfir.designate %[[DECL]]#0 (%[[IV0]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %{{.*}} to %[[ADDR0]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// Second loop: read back from the temp using the new induction variable.
// CHECK:           fir.do_loop %[[IV1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             fir.if %{{.*}} {
// CHECK:               %[[ADDR1:.*]] = hlfir.designate %[[DECL]]#0 (%[[IV1]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL:.*]] = fir.load %[[ADDR1]] : !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL]] to %{{.*}} : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return

// Two nested forall loops with non-trivial lower bounds: the temp is a 2D
// ArrayTemp whose first dimension corresponds to the inner forall and second
// dimension to the outer one. The fir.shape_shift carries the forall lower
// bounds, and hlfir.designate uses both induction variables (innermost first).
func.func @nested_forall_2d(%arr: !fir.box<!fir.array<?x?xi32>>) {
  %c2_i32 = arith.constant 2 : i32
  %c5_i32 = arith.constant 5 : i32
  %c3_i32 = arith.constant 3 : i32
  %c7_i32 = arith.constant 7 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:2 = hlfir.declare %arr {uniq_name = "x"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  hlfir.forall lb {
    hlfir.yield %c2_i32 : i32
  } ub {
    hlfir.yield %c5_i32 : i32
  }  (%i: i32) {
    hlfir.forall lb {
      hlfir.yield %c3_i32 : i32
    } ub {
      hlfir.yield %c7_i32 : i32
    }  (%j: i32) {
      hlfir.region_assign {
        %i_idx = fir.convert %i : (i32) -> i64
        %j_idx = fir.convert %j : (i32) -> i64
        %addr = hlfir.designate %0#0 (%i_idx, %j_idx)  : (!fir.box<!fir.array<?x?xi32>>, i64, i64) -> !fir.ref<i32>
        %val = fir.load %addr : !fir.ref<i32>
        hlfir.yield %val : i32
      } to {
        %i_inv = arith.subi %c5_i32, %i : i32
        %i_inv2 = arith.addi %i_inv, %c2_i32 : i32
        %j_inv = arith.subi %c7_i32, %j : i32
        %j_inv2 = arith.addi %j_inv, %c3_i32 : i32
        %i_idx = fir.convert %i_inv2 : (i32) -> i64
        %j_idx = fir.convert %j_inv2 : (i32) -> i64
        %addr = hlfir.designate %0#0 (%i_idx, %j_idx)  : (!fir.box<!fir.array<?x?xi32>>, i64, i64) -> !fir.ref<i32>
        hlfir.yield %addr : !fir.ref<i32>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @nested_forall_2d(
// CHECK-NOT:       fir.alloca index
// CHECK:           %[[ALLOC:.*]] = fir.allocmem !fir.array<?x?xi32>, %{{.*}}, %{{.*}} {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[SHAPE:.*]] = fir.shape_shift %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?x?xi32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.heap<!fir.array<?x?xi32>>)
// CHECK:           fir.do_loop %[[I0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             fir.do_loop %[[J0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[ADDR0:.*]] = hlfir.designate %[[DECL]]#0 (%[[J0]], %[[I0]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %{{.*}} to %[[ADDR0]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.do_loop %[[I1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             fir.do_loop %[[J1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[ADDR1:.*]] = hlfir.designate %[[DECL]]#0 (%[[J1]], %[[I1]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[V:.*]] = fir.load %[[ADDR1]] : !fir.ref<i32>
// CHECK:               hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?x?xi32>>
// CHECK:           return
