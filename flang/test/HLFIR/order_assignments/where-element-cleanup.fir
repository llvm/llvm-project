// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

// Applying the same producer twice recursively inlines two distinct scalar
// temporaries. Each cleanup must keep the mapping from its own evaluation.
func.func @repeated_apply(%mask: !fir.ref<!fir.array<10x!fir.logical<4>>>, %a: !fir.ref<!fir.array<10x!fir.char<1,2>>>, %out: !fir.ref<!fir.array<10x!fir.char<1,8>>>) {
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.region_assign {
      %producer = hlfir.elemental %shape typeparams %c4 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,4>> {
      ^bb0(%i: index):
        %a_i = hlfir.designate %a (%i) typeparams %c2 : (!fir.ref<!fir.array<10x!fir.char<1,2>>>, index, index) -> !fir.ref<!fir.char<1,2>>
        %temp = hlfir.concat %a_i, %a_i len %c4 : (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, index) -> !hlfir.expr<!fir.char<1,4>>
        hlfir.yield_element %temp : !hlfir.expr<!fir.char<1,4>> cleanup {
          hlfir.destroy %temp : !hlfir.expr<!fir.char<1,4>>
        }
      }
      %consumer = hlfir.elemental %shape typeparams %c8 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,8>> {
      ^bb0(%j: index):
        %left = hlfir.apply %producer, %j typeparams %c4 : (!hlfir.expr<10x!fir.char<1,4>>, index, index) -> !hlfir.expr<!fir.char<1,4>>
        %right = hlfir.apply %producer, %j typeparams %c4 : (!hlfir.expr<10x!fir.char<1,4>>, index, index) -> !hlfir.expr<!fir.char<1,4>>
        %joined = hlfir.concat %left, %right len %c8 : (!hlfir.expr<!fir.char<1,4>>, !hlfir.expr<!fir.char<1,4>>, index) -> !hlfir.expr<!fir.char<1,8>>
        hlfir.yield_element %joined : !hlfir.expr<!fir.char<1,8>> cleanup {
          hlfir.destroy %joined : !hlfir.expr<!fir.char<1,8>>
        }
      }
      hlfir.yield %consumer : !hlfir.expr<10x!fir.char<1,8>> cleanup {
        hlfir.destroy %consumer : !hlfir.expr<10x!fir.char<1,8>>
        hlfir.destroy %producer : !hlfir.expr<10x!fir.char<1,4>>
      }
    } to {
      hlfir.yield %out : !fir.ref<!fir.array<10x!fir.char<1,8>>>
    }
  }
  return
}
// CHECK-LABEL: func.func @repeated_apply
// CHECK: fir.do_loop
// CHECK: fir.if
// CHECK: %[[LEFT:.*]] = hlfir.concat
// CHECK: %[[RIGHT:.*]] = hlfir.concat
// CHECK: %[[JOINED:.*]] = hlfir.concat %[[LEFT]], %[[RIGHT]]
// CHECK: hlfir.assign %[[JOINED]]
// CHECK-NEXT: hlfir.destroy %[[JOINED]]
// CHECK-NEXT: hlfir.destroy %[[RIGHT]]
// CHECK-NEXT: hlfir.destroy %[[LEFT]]
// CHECK-NOT: hlfir.destroy
// CHECK: return
