// The IV recurrence's result (%ivNext) has body uses beyond the
// scf.yield — here a `memref.store`, modelling the `fir.declare_value` ops
// mem2reg emits after promoting a Fortran scalar variable. The body-clone
// loop walks the continues block in source order; the IV recurrence is
// cloned as a regular body op (its operand %iv resolves to ivAsOrig via
// the IRMapping, and the loop-invariant step passes through unmapped).
// SSA ordering guarantees the clone is in the new body before any cloned
// user of its result. Without the mapping, the cloned `memref.store` would
// hold a dangling reference to the original %ivNext inside the erased
// scf.while.

// RUN: fir-opt %s --lift-scf-while-to-scf-for 2>/dev/null | FileCheck %s

func.func @iv_next_with_body_use(%lb: i32, %N: i32, %sink: memref<i32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %poison = arith.constant 0 : i32
  %r:2 = scf.while (%trip = %N, %iv = %lb) : (i32, i32) -> (i32, i32) {
    %cmp = arith.cmpi sgt, %trip, %c0 : i32
    %ifr:3 = scf.if %cmp -> (i32, i32, i32) {
      %tripNext = arith.subi %trip, %c1 : i32
      %ivNext   = arith.addi %iv, %c1 : i32
      // %ivNext is consumed both by the scf.yield AND by this store;
      // the latter survives cloning and must point at the new ivNextInFor.
      memref.store %ivNext, %sink[] : memref<i32>
      scf.yield %tripNext, %ivNext, %c1 : i32, i32, i32
    } else {
      scf.yield %poison, %poison, %c0 : i32, i32, i32
    }
    %enter = arith.trunci %ifr#2 : i32 to i1
    scf.condition(%enter) %ifr#0, %ifr#1 : i32, i32
  } do {
  ^bb0(%a0: i32, %a1: i32):
    scf.yield %a0, %a1 : i32, i32
  }
  return
}

// CHECK-LABEL: func.func @iv_next_with_body_use(
// CHECK-SAME:      %{{.*}}: i32, %[[N:.*]]: i32, %[[SINK:.*]]: memref<i32>)

// CHECK-NOT:   scf.while

// Inside the new scf.for body, the IV is cast back to i32 (this is the
// `ivAsOrig` value), then the IV-recurrence equivalent is materialized as
// `arith.addi %ivAsOrig, %c1`. The cloned `memref.store` must reference
// THAT materialized value, NOT a dangling pointer into the destroyed
// scf.if.
//
// CHECK:       scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:         %[[IVi32:.*]]    = arith.index_cast %[[I]] : index to i32
// CHECK:         %[[IVNEXT:.*]]   = arith.addi %[[IVi32]], %{{.*}} : i32
// CHECK:         memref.store %[[IVNEXT]], %[[SINK]][] : memref<i32>
// CHECK:       }
