// Counted scf.while with ivStep == 1.
//
// The pass should emit the upper bound as `arith.addi ivInit, tripInit` —
// no `arith.muli`, since multiplying tripInit by 1 is a no-op.

// RUN: fir-opt %s --lift-scf-while-to-scf-for 2>/dev/null | FileCheck %s

func.func @counted_step1(%lb: i32, %N: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %poison = arith.constant 0 : i32
  %r:2 = scf.while (%trip = %N, %iv = %lb) : (i32, i32) -> (i32, i32) {
    %cmp = arith.cmpi sgt, %trip, %c0 : i32
    %ifr:3 = scf.if %cmp -> (i32, i32, i32) {
      %tripNext = arith.subi %trip, %c1 : i32
      %ivNext   = arith.addi %iv,  %c1 : i32
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

// CHECK-LABEL: func.func @counted_step1(
// CHECK-SAME:      %[[LB:.*]]: i32, %[[N:.*]]: i32)

// scf.while is gone, scf.for takes over.
// CHECK-NOT:     scf.while

// ub = lb + N (no muli, since ivStep == 1)
// CHECK-NOT:     arith.muli
// CHECK:         %[[UB:.*]] = arith.addi %[[LB]], %[[N]] : i32

// lb, ub and step cast to index for the scf.for.
// CHECK:         %[[LB_IDX:.*]]   = arith.index_cast %[[LB]] : i32 to index
// CHECK:         %[[UB_IDX:.*]]   = arith.index_cast %[[UB]] : i32 to index
// CHECK:         %[[STEP_IDX:.*]] = arith.index_cast %{{.*}} : i32 to index

// scf.for over that range; no muli inside the body either.
// CHECK:         scf.for %{{.*}} = %[[LB_IDX]] to %[[UB_IDX]] step %[[STEP_IDX]] {
// CHECK-NOT:       arith.muli
// CHECK:         }
