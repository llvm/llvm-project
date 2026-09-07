// Counted scf.while with one induction variable and two pass-through
// iter_args. Verifies that:
//   * the rewrite preserves both iter_args in order, threading their inits
//     from the scf.while inits and their next-iteration values from the
//     correct contYield slots,
//   * uses of the original iter_arg before-args inside the body are
//     remapped to the corresponding scf.for iter_arg block arguments,
//   * the final scf.yield carries the cloned next-values for both
//     iter_args.

// RUN: fir-opt %s --lift-scf-while-to-scf-for 2>/dev/null | FileCheck %s

func.func @two_iter(%lb: i32, %N: i32, %x0: i32, %y0: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %poison = arith.constant 0 : i32
  %r:4 = scf.while (%trip = %N, %iv = %lb, %x = %x0, %y = %y0)
        : (i32, i32, i32, i32) -> (i32, i32, i32, i32) {
    %cmp = arith.cmpi sgt, %trip, %c0 : i32
    %ifr:5 = scf.if %cmp -> (i32, i32, i32, i32, i32) {
      %tripNext = arith.subi %trip, %c1 : i32
      %ivNext   = arith.addi %iv,  %c1 : i32
      // non-affine recurrence in %x (rhs is the IV, not a loop invariant)
      %xNext    = arith.muli %x,   %iv : i32
      // non-affine recurrence in %y (subi where rhs is the IV, not invariant)
      %yNext    = arith.subi %y,   %iv : i32
      scf.yield %tripNext, %ivNext, %xNext, %yNext, %c1
        : i32, i32, i32, i32, i32
    } else {
      scf.yield %poison, %poison, %poison, %poison, %c0
        : i32, i32, i32, i32, i32
    }
    %enter = arith.trunci %ifr#4 : i32 to i1
    scf.condition(%enter) %ifr#0, %ifr#1, %ifr#2, %ifr#3 : i32, i32, i32, i32
  } do {
  ^bb0(%a0: i32, %a1: i32, %a2: i32, %a3: i32):
    scf.yield %a0, %a1, %a2, %a3 : i32, i32, i32, i32
  }
  return
}

// CHECK-LABEL: func.func @two_iter(
// CHECK-SAME:      %[[LB:.*]]: i32, %[[N:.*]]: i32, %[[X0:.*]]: i32, %[[Y0:.*]]: i32)

// scf.while is replaced by scf.for.
// CHECK-NOT:   scf.while

// Bounds materialization (ivStep == 1, so no muli; just `lb + N`).
// CHECK-NOT:     arith.muli {{%.*}}, {{%c1}}
// CHECK:         %[[UB:.*]] = arith.addi %[[LB]], %[[N]] : i32

// Index-typed bounds.
// CHECK:         %[[LB_IDX:.*]] = arith.index_cast %[[LB]]
// CHECK:         %[[UB_IDX:.*]] = arith.index_cast %[[UB]]
// CHECK:         %[[STEP_IDX:.*]] = arith.index_cast %{{.*}}

// scf.for with two iter_args, in the same order as the source's non-IV
// before-args. The first iter_arg is initialized from %x0, the second
// from %y0.
// CHECK:         scf.for %[[I:.*]] = %[[LB_IDX]] to %[[UB_IDX]] step %[[STEP_IDX]]
// CHECK-SAME:        iter_args(%[[XARG:.*]] = %[[X0]], %[[YARG:.*]] = %[[Y0]])
// CHECK-SAME:        -> (i32, i32)

// Body: cast IV back to i32 and recompute the two non-affine next-values
// using the iter_arg block arguments (not the original SSA names).
// CHECK:           %[[I_I32:.*]] = arith.index_cast %[[I]] : index to i32
// CHECK:           %[[XNEXT:.*]] = arith.muli %[[XARG]], %[[I_I32]] : i32
// CHECK:           %[[YNEXT:.*]] = arith.subi %[[YARG]], %[[I_I32]] : i32

// Yield carries the next-iter values in the same iter_args order.
// CHECK:           scf.yield %[[XNEXT]], %[[YNEXT]] : i32, i32
// CHECK:         }
