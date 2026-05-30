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

// Forall with a negative unit step. The temp's lower bound must be the
// loop's upper bound (the smaller end of the iv range), not the loop's
// lower bound (which is the start, i.e. the largest iv value); otherwise
// the loop induction variable falls below the temp's domain on every
// iteration but the first.
func.func @forall_negative_step(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  hlfir.forall lb {
    hlfir.yield %c10 : index
  } ub {
    hlfir.yield %c1 : index
  } step {
    hlfir.yield %c-1 : index
  }  (%i: index) {
    hlfir.region_assign {
      %rev = arith.subi %c11, %i : index
      %addr = hlfir.designate %1#0 (%rev)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %val = fir.load %addr : !fir.ref<i32>
      hlfir.yield %val : i32
    } to {
      %addr = hlfir.designate %1#0 (%i)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      hlfir.yield %addr : !fir.ref<i32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @forall_negative_step(
// CHECK-NOT:       fir.alloca index
// The do_loop runs from the loop's lb (c10) down to its ub (c1) with step
// c-1. The temp's lower bound is the loop's upper bound (c1), not its
// lower bound, so all iv values [1..10] fit within the temp's domain.
// CHECK:           %[[C_NEG1:.*]] = arith.constant -1 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C10:.*]] = arith.constant 10 : index
// CHECK:           %[[ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[SHAPE:.*]] = fir.shape_shift %[[C1]], %{{.*}} : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.do_loop %[[IV0:.*]] = %[[C10]] to %[[C1]] step %[[C_NEG1]] {
// CHECK:             %[[ADDR0:.*]] = hlfir.designate %[[DECL]]#0 (%[[IV0]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %{{.*}} to %[[ADDR0]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           fir.do_loop %[[IV1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[ADDR1:.*]] = hlfir.designate %[[DECL]]#0 (%[[IV1]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[V:.*]] = fir.load %[[ADDR1]] : !fir.ref<i32>
// CHECK:             hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?xi32>>
// CHECK:           return

// Forall with a non-unit step (step=2): the loop iv values are
// non-contiguous (1, 3, 5, 7, 9), so the loop induction variable cannot
// directly index a contiguous fir.shape_shift temp. We must fall back to
// the 1D counter-based HomogeneousScalarStack.
func.func @forall_non_unit_step(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %c11 = arith.constant 11 : index
  %0 = fir.shape %c9 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "x"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c9 : index
  } step {
    hlfir.yield %c2 : index
  }  (%i: index) {
    hlfir.region_assign {
      %rev = arith.subi %c11, %i : index
      %addr = hlfir.designate %1#0 (%rev)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %val = fir.load %addr : !fir.ref<i32>
      hlfir.yield %val : i32
    } to {
      %addr = hlfir.designate %1#0 (%i)  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      hlfir.yield %addr : !fir.ref<i32>
    }
  }
  return
}
// CHECK-LABEL:   func.func @forall_non_unit_step(
// Counter-based HomogeneousScalarStack: a fir.alloca index counter and a
// plain fir.shape (no shift), with the temp indexed through the counter.
// CHECK:           %[[CTR:.*]] = fir.alloca index
// CHECK:           %[[ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp.forall", uniq_name = ""}
// CHECK:           %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
// CHECK:           hlfir.declare %[[ALLOC]](%[[SHAPE]]) {uniq_name = ".tmp.forall"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK:           fir.load %[[CTR]] : !fir.ref<index>
// CHECK:           arith.addi %{{.*}}, %{{.*}} : index
// CHECK:           fir.store %{{.*}} to %[[CTR]] : !fir.ref<index>
