// RUN: fir-opt %s --canonicalize | FileCheck %s

// Reboxing a guarded box address must not fold back to the original box.
// CHECK-LABEL: func.func @guarded_box_address(
// CHECK: %[[ADDR:.*]] = fir.box_addr
// CHECK: %[[GUARDED:.*]] = fir.assert {{.*}} guard(%[[ADDR]]
// CHECK: %[[BOX:.*]] = fir.embox %[[GUARDED]]
// CHECK: return %[[BOX]]
func.func @guarded_box_address(%box: !fir.box<!fir.array<10xi32>>, %ok: i1) -> !fir.box<!fir.array<10xi32>> {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %addr = fir.box_addr %box : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  %g = fir.assert %ok, pointer, "invalid pointer" guard(%addr : !fir.ref<!fir.array<10xi32>>)
  %result = fir.embox %g(%shape) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  return %result : !fir.box<!fir.array<10xi32>>
}
