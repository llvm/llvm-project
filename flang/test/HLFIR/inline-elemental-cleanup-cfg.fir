// RUN: fir-opt %s --inline-elementals | FileCheck %s

// Conservatively keep the array even when cleanup is in an unconditional
// successor block. Scalar cleanup is only supported in the apply block.
func.func @successor_cleanup(%a: !fir.ref<!fir.array<10x!fir.char<1,2>>>, %out: !fir.ref<!fir.char<1,4>>, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %shape typeparams %c4 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,4>> {
  ^bb0(%i: index):
    %a_i = hlfir.designate %a (%i) typeparams %c2 : (!fir.ref<!fir.array<10x!fir.char<1,2>>>, index, index) -> !fir.ref<!fir.char<1,2>>
    %temp = hlfir.concat %a_i, %a_i len %c4 : (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, index) -> !hlfir.expr<!fir.char<1,4>>
    hlfir.yield_element %temp : !hlfir.expr<!fir.char<1,4>> cleanup {
      hlfir.destroy %temp : !hlfir.expr<!fir.char<1,4>>
    }
  }

  %element = hlfir.apply %expr, %c1 typeparams %c4 : (!hlfir.expr<10x!fir.char<1,4>>, index, index) -> !hlfir.expr<!fir.char<1,4>>
  hlfir.assign %element to %out : !hlfir.expr<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
  cf.br ^cleanup
^cleanup:
  hlfir.destroy %expr : !hlfir.expr<10x!fir.char<1,4>>
  return
}
// CHECK-LABEL: func.func @successor_cleanup
// CHECK: %[[ARRAY:.*]] = hlfir.elemental
// CHECK: hlfir.apply %[[ARRAY]]
// CHECK: hlfir.assign
// CHECK-NEXT: cf.br ^[[EXIT:.*]]
// CHECK: ^[[EXIT]]:
// CHECK-NEXT: hlfir.destroy %[[ARRAY]]
// CHECK-NEXT: return

// The array dominates the join, but a scalar from only one branch does not.
func.func @conditional_apply(%a: !fir.ref<!fir.array<10x!fir.char<1,2>>>, %out: !fir.ref<!fir.char<1,4>>, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %shape typeparams %c4 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,4>> {
  ^bb0(%i: index):
    %a_i = hlfir.designate %a (%i) typeparams %c2 : (!fir.ref<!fir.array<10x!fir.char<1,2>>>, index, index) -> !fir.ref<!fir.char<1,2>>
    %temp = hlfir.concat %a_i, %a_i len %c4 : (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, index) -> !hlfir.expr<!fir.char<1,4>>
    hlfir.yield_element %temp : !hlfir.expr<!fir.char<1,4>> cleanup {
      hlfir.destroy %temp : !hlfir.expr<!fir.char<1,4>>
    }
  }
  cf.cond_br %condition, ^then, ^cleanup
^then:
  %element = hlfir.apply %expr, %c1 typeparams %c4 : (!hlfir.expr<10x!fir.char<1,4>>, index, index) -> !hlfir.expr<!fir.char<1,4>>
  hlfir.assign %element to %out : !hlfir.expr<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
  cf.br ^cleanup
^cleanup:
  hlfir.destroy %expr : !hlfir.expr<10x!fir.char<1,4>>
  return
}
// CHECK-LABEL: func.func @conditional_apply
// CHECK: %[[ARRAY:.*]] = hlfir.elemental
// CHECK: cf.cond_br
// CHECK: hlfir.apply %[[ARRAY]]
// CHECK: hlfir.assign
// CHECK: cf.br
// CHECK: hlfir.destroy %[[ARRAY]]
// CHECK: return

// A scalar produced inside a loop cannot be cleaned up outside its region.
func.func @loop_apply(%a: !fir.ref<!fir.array<10x!fir.char<1,2>>>, %out: !fir.ref<!fir.char<1,4>>, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %shape typeparams %c4 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,4>> {
  ^bb0(%i: index):
    %a_i = hlfir.designate %a (%i) typeparams %c2 : (!fir.ref<!fir.array<10x!fir.char<1,2>>>, index, index) -> !fir.ref<!fir.char<1,2>>
    %temp = hlfir.concat %a_i, %a_i len %c4 : (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, index) -> !hlfir.expr<!fir.char<1,4>>
    hlfir.yield_element %temp : !hlfir.expr<!fir.char<1,4>> cleanup {
      hlfir.destroy %temp : !hlfir.expr<!fir.char<1,4>>
    }
  }
  fir.do_loop %j = %c1 to %c10 step %c1 {
    %element = hlfir.apply %expr, %c1 typeparams %c4 : (!hlfir.expr<10x!fir.char<1,4>>, index, index) -> !hlfir.expr<!fir.char<1,4>>
    hlfir.assign %element to %out : !hlfir.expr<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
  }
  hlfir.destroy %expr : !hlfir.expr<10x!fir.char<1,4>>
  return
}
// CHECK-LABEL: func.func @loop_apply
// CHECK: %[[ARRAY:.*]] = hlfir.elemental
// CHECK: fir.do_loop
// CHECK: hlfir.apply %[[ARRAY]]
// CHECK: hlfir.assign
// CHECK: }
// CHECK-NEXT: hlfir.destroy %[[ARRAY]]
// CHECK-NEXT: return

// Apply dominates destroy, but can execute several times before it. Cleaning
// up only at destroy would leak all but the last scalar temporary.
func.func @cfg_loop_apply(%a: !fir.ref<!fir.array<10x!fir.char<1,2>>>, %out: !fir.ref<!fir.char<1,4>>, %condition: i1) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %shape typeparams %c4 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,4>> {
  ^bb0(%i: index):
    %a_i = hlfir.designate %a (%i) typeparams %c2 : (!fir.ref<!fir.array<10x!fir.char<1,2>>>, index, index) -> !fir.ref<!fir.char<1,2>>
    %temp = hlfir.concat %a_i, %a_i len %c4 : (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, index) -> !hlfir.expr<!fir.char<1,4>>
    hlfir.yield_element %temp : !hlfir.expr<!fir.char<1,4>> cleanup {
      hlfir.destroy %temp : !hlfir.expr<!fir.char<1,4>>
    }
  }
  cf.br ^loop
^loop:
  %element = hlfir.apply %expr, %c1 typeparams %c4 : (!hlfir.expr<10x!fir.char<1,4>>, index, index) -> !hlfir.expr<!fir.char<1,4>>
  hlfir.assign %element to %out : !hlfir.expr<!fir.char<1,4>>, !fir.ref<!fir.char<1,4>>
  cf.cond_br %condition, ^loop, ^cleanup
^cleanup:
  hlfir.destroy %expr : !hlfir.expr<10x!fir.char<1,4>>
  return
}
// CHECK-LABEL: func.func @cfg_loop_apply
// CHECK: %[[ARRAY:.*]] = hlfir.elemental
// CHECK: cf.br
// CHECK: hlfir.apply %[[ARRAY]]
// CHECK: hlfir.assign
// CHECK: cf.cond_br
// CHECK: hlfir.destroy %[[ARRAY]]
// CHECK-NEXT: return
