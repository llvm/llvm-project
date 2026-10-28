// Test code generation of hlfir.where when an inlined hlfir.elemental yields
// a variable (e.g. the element of a vector subscripted designator) and is
// applied in a context that requires a value.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

// where (msk) de = (dd(idx))
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
        hlfir.yield_element %5 : !fir.ref<!fir.type<t{v:i32}>>
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
// CHECK:               %[[IDXADDR:.*]] = hlfir.designate %{{.*}} (%[[I]])  : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
// CHECK:               %[[IDXVAL:.*]] = fir.load %[[IDXADDR]] : !fir.ref<i64>
// CHECK:               %[[ELT:.*]] = hlfir.designate %{{.*}} (%[[IDXVAL]])  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, i64) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               %[[VAL:.*]] = hlfir.as_expr %[[ELT]] : (!fir.ref<!fir.type<t{v:i32}>>) -> !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK:               %[[PAREN:.*]] = hlfir.no_reassoc %[[VAL]] : !hlfir.expr<!fir.type<t{v:i32}>>
// CHECK:               %[[LHS:.*]] = hlfir.designate %{{.*}} (%[[I]])  : (!fir.ref<!fir.array<3x!fir.type<t{v:i32}>>>, index) -> !fir.ref<!fir.type<t{v:i32}>>
// CHECK:               hlfir.assign %[[PAREN]] to %[[LHS]] : !hlfir.expr<!fir.type<t{v:i32}>>, !fir.ref<!fir.type<t{v:i32}>>
// CHECK-NOT:       hlfir.destroy


// An hlfir.elemental yielding the address of a trivial element (lowering
// currently yields the loaded value instead) must be loaded when applied.
func.func @trivial_add(%arg0: !fir.ref<!fir.array<3xi32>>, %arg1: !fir.ref<!fir.array<3xi32>>, %arg2: !fir.ref<!fir.array<3xi64>>, %arg3: !fir.ref<!fir.array<3x!fir.logical<4>>>) {
  %c3 = arith.constant 3 : index
  %0 = fir.shape %c3 : (index) -> !fir.shape<1>
  %x:2 = hlfir.declare %arg0(%0) uniq_name("x") : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %y:2 = hlfir.declare %arg1(%0) uniq_name("y") : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %idx:2 = hlfir.declare %arg2(%0) uniq_name("idx") : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
  %msk:2 = hlfir.declare %arg3(%0) uniq_name("msk") : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
  hlfir.where {
    hlfir.yield %msk#0 : !fir.ref<!fir.array<3x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %1 = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
      ^bb0(%i: index):
        %3 = hlfir.designate %idx#0 (%i) : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
        %4 = fir.load %3 : !fir.ref<i64>
        %5 = hlfir.designate %x#0 (%4) : (!fir.ref<!fir.array<3xi32>>, i64) -> !fir.ref<i32>
        hlfir.yield_element %5 : !fir.ref<i32>
      }
      %2 = hlfir.elemental %0 unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32> {
      ^bb0(%i: index):
        %3 = hlfir.apply %1, %i : (!hlfir.expr<3xi32>, index) -> i32
        %c1 = arith.constant 1 : i32
        %4 = arith.addi %3, %c1 : i32
        hlfir.yield_element %4 : i32
      }
      hlfir.yield %2 : !hlfir.expr<3xi32> cleanup {
        hlfir.destroy %2 : !hlfir.expr<3xi32>
        hlfir.destroy %1 : !hlfir.expr<3xi32>
      }
    } to {
      hlfir.yield %y#0 : !fir.ref<!fir.array<3xi32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @trivial_add(
// CHECK-NOT:       hlfir.elemental
// CHECK:           fir.do_loop %[[I:.*]] =
// CHECK:             fir.if
// CHECK:               %[[ELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<3xi32>>, i64) -> !fir.ref<i32>
// CHECK-NOT:           hlfir.as_expr
// CHECK:               %[[VAL:.*]] = fir.load %[[ELT]] : !fir.ref<i32>
// CHECK:               %[[ADD:.*]] = arith.addi %[[VAL]], %{{.*}} : i32
// CHECK:               %[[LHS:.*]] = hlfir.designate %{{.*}} (%[[I]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[ADD]] to %[[LHS]] : i32, !fir.ref<i32>
