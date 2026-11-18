// Test code generation of hlfir.where when an inlined hlfir.elemental yields
// an hlfir.as_expr copy of a variable element (e.g. the element of a vector
// subscripted designator).
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

// where (msk) de = dd(idx)
func.func @derived_assign(%arg0: !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, %arg1: !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, %arg2: !fir.ref<!fir.array<3xi64>>, %arg3: !fir.ref<!fir.array<3x!fir.logical<4>>>) {
  %c3 = arith.constant 3 : index
  %0 = fir.shape %c3 : (index) -> !fir.shape<1>
  %dd:2 = hlfir.declare %arg0(%0) uniq_name("dd") : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>)
  %de:2 = hlfir.declare %arg1(%0) uniq_name("de") : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>)
  %idx:2 = hlfir.declare %arg2(%0) uniq_name("idx") : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
  %msk:2 = hlfir.declare %arg3(%0) uniq_name("msk") : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %msk#0 : !fir.ref<!fir.array<3x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %1 = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<t{v:i32}>> {
      ^bb0(%i: index):
        %3 = hlfir.designate %idx#0 (%i) : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
        %4 = fir.load %3 : !fir.ref<i64>
        %5 = hlfir.designate %dd#0 (%4) : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
        %6 = hlfir.as_expr %5 : (!fir.ref<!fir.type<t{v:i32}>>) -> !hlfir.expr<!fir.type<t{v:i32}>>
        hlfir.yield_element %6 : !hlfir.expr<!fir.type<t{v:i32}>> cleanup {
          hlfir.destroy %6 : !hlfir.expr<!fir.type<t{v:i32}>>
        }
      }
      hlfir.yield %1 : !hlfir.expr<3x!fir.type<t{v:i32}>> cleanup {
        hlfir.destroy %1 : !hlfir.expr<3x!fir.type<t{v:i32}>>
      }
    } to {
      hlfir.yield %de#0 : !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @derived_assign(
// CHECK-NOT:       hlfir.elemental
// CHECK:           fir.do_loop %[[I:.*]] =
// CHECK:             fir.if
// CHECK:               %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK-NOT:           hlfir.as_expr
// CHECK:               %[[LHS:.*]] = hlfir.designate %{{.*}} (%[[I]])  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, index) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               hlfir.assign %[[ELT]] to %[[LHS]] : !fir.ref<!fir.type<t{v:i32}>>, !fir.ref<!fir.type<t{v:i32}>>
// CHECK-NOT:           hlfir.destroy %{{.*}} : !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK-NOT:       hlfir.destroy %{{.*}} : !hlfir.expr<3x

// where (msk) de = (dd(idx)): the hlfir.as_expr is needed by hlfir.no_reassoc.
func.func @derived_parens(%arg0: !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, %arg1: !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, %arg2: !fir.ref<!fir.array<3xi64>>, %arg3: !fir.ref<!fir.array<3x!fir.logical<4>>>) {
  %c3 = arith.constant 3 : index
  %0 = fir.shape %c3 : (index) -> !fir.shape<1>
  %dd:2 = hlfir.declare %arg0(%0) uniq_name("dd") : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>)
  %de:2 = hlfir.declare %arg1(%0) uniq_name("de") : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>)
  %idx:2 = hlfir.declare %arg2(%0) uniq_name("idx") : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
  %msk:2 = hlfir.declare %arg3(%0) uniq_name("msk") : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %msk#0 : !fir.ref<!fir.array<3x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %1 = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<t{v:i32}>> {
      ^bb0(%i: index):
        %3 = hlfir.designate %idx#0 (%i) : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
        %4 = fir.load %3 : !fir.ref<i64>
        %5 = hlfir.designate %dd#0 (%4) : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
        %6 = hlfir.as_expr %5 : (!fir.ref<!fir.type<t{v:i32}>>) -> !hlfir.expr<!fir.type<t{v:i32}>>
        hlfir.yield_element %6 : !hlfir.expr<!fir.type<t{v:i32}>> cleanup {
          hlfir.destroy %6 : !hlfir.expr<!fir.type<t{v:i32}>>
        }
      }
      %2 = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<t{v:i32}>> {
      ^bb0(%i: index):
        %3 = hlfir.apply %1, %i : (!hlfir.expr<3x!fir.type<t{v:i32}>>, index) -> !hlfir.expr<!fir.type<t{v:i32}>>
        %4 = hlfir.no_reassoc %3 : !hlfir.expr<!fir.type<t{v:i32}>>
        hlfir.yield_element %4 : !hlfir.expr<!fir.type<t{v:i32}>>
      }
      hlfir.yield %2 : !hlfir.expr<3x!fir.type<t{v:i32}>> cleanup {
        hlfir.destroy %2 : !hlfir.expr<3x!fir.type<t{v:i32}>>
        hlfir.destroy %1 : !hlfir.expr<3x!fir.type<t{v:i32}>>
      }
    } to {
      hlfir.yield %de#0 : !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @derived_parens(
// CHECK-NOT:       hlfir.elemental
// CHECK:           fir.do_loop %[[I:.*]] =
// CHECK:             fir.if
// CHECK:               %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<t{v:i32}>>) -> !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK:               %[[PAREN:.*]] = hlfir.no_reassoc %[[VAL]] : !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK:               %[[LHS:.*]] = hlfir.designate %{{.*}} (%[[I]])  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, index) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               hlfir.assign %[[PAREN]] to %[[LHS]] : !hlfir.expr<!fir.type<t{v:i32}>>, !fir.ref<!fir.type<t{v:i32}>>
// CHECK-NEXT:          hlfir.destroy %[[VAL]] : !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK-NOT:       hlfir.destroy %{{.*}} : !hlfir.expr<3x

// The hlfir.as_expr copy is kept if any operation follows it in the elemental
// body, since such operations (e.g. clean-ups) could modify or end the lifetime
// of the copied variable before the assignment.
func.func @derived_op_after_as_expr(%arg0: !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, %arg1: !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, %arg2: !fir.ref<!fir.array<3xi64>>, %arg3: !fir.ref<!fir.array<3x!fir.logical<4>>>) {
  %c3 = arith.constant 3 : index
  %0 = fir.shape %c3 : (index) -> !fir.shape<1>
  %dd:2 = hlfir.declare %arg0(%0) uniq_name("dd") : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>)
  %de:2 = hlfir.declare %arg1(%0) uniq_name("de") : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>)
  %idx:2 = hlfir.declare %arg2(%0) uniq_name("idx") : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
  %msk:2 = hlfir.declare %arg3(%0) uniq_name("msk") : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %msk#0 : !fir.ref<!fir.array<3x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %1 = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.type<t{v:i32}>> {
      ^bb0(%i: index):
        %3 = hlfir.designate %idx#0 (%i) : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
        %4 = fir.load %3 : !fir.ref<i64>
        %5 = hlfir.designate %dd#0 (%4) : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
        %6 = hlfir.as_expr %5 : (!fir.ref<!fir.type<t{v:i32}>>) -> !hlfir.expr<!fir.type<t{v:i32}>>
        %7 = fir.load %3 : !fir.ref<i64>
        hlfir.yield_element %6 : !hlfir.expr<!fir.type<t{v:i32}>> cleanup {
          hlfir.destroy %6 : !hlfir.expr<!fir.type<t{v:i32}>>
        }
      }
      hlfir.yield %1 : !hlfir.expr<3x!fir.type<t{v:i32}>> cleanup {
        hlfir.destroy %1 : !hlfir.expr<3x!fir.type<t{v:i32}>>
      }
    } to {
      hlfir.yield %de#0 : !fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @derived_op_after_as_expr(
// CHECK-NOT:       hlfir.elemental
// CHECK:           fir.do_loop %[[I:.*]] =
// CHECK:             fir.if
// CHECK:               %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<t{v:i32}>>) -> !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK:               fir.load
// CHECK:               %[[LHS:.*]] = hlfir.designate %{{.*}} (%[[I]])  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, index) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               hlfir.assign %[[VAL]] to %[[LHS]] : !hlfir.expr<!fir.type<t{v:i32}>>, !fir.ref<!fir.type<t{v:i32}>>
// CHECK-NEXT:          hlfir.destroy %[[VAL]] : !hlfir.expr<!fir.type<t{v:i32}>>
