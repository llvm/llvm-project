// Test code generation of hlfir.where/hflir.elsewhere when an
// "impure" mask is used and several runs are needed. The mask
// must be saved so that the impure function is only evaluated once.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func private @impure() -> !fir.heap<!fir.array<10x!fir.logical<4>>>
func.func @test_elsewhere_impure_mask(%x: !fir.ref<!fir.array<10xi32>>, %y: !fir.ref<!fir.array<10xi32>>, %z: !fir.ref<!fir.array<10xi32>>, %mask: !fir.ref<!fir.array<10x!fir.logical<4>>>) {
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  hlfir.where {
    hlfir.yield %mask : !fir.ref<!fir.array<10x!fir.logical<4>>>
  } do {
    hlfir.elsewhere mask {
      %mask2 = fir.call @impure() : () -> !fir.heap<!fir.array<10x!fir.logical<4>>>
      hlfir.yield %mask2 : !fir.heap<!fir.array<10x!fir.logical<4>>> cleanup {
        fir.freemem %mask2 : !fir.heap<!fir.array<10x!fir.logical<4>>>
      }
    } do {
      hlfir.region_assign {
        hlfir.yield %y : !fir.ref<!fir.array<10xi32>>
      } to {
        hlfir.yield %x : !fir.ref<!fir.array<10xi32>>
      }
      hlfir.region_assign {
        hlfir.yield %x : !fir.ref<!fir.array<10xi32>>
      } to {
        hlfir.yield %z : !fir.ref<!fir.array<10xi32>>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_elsewhere_impure_mask(
// CHECK:           %[[VAL_12:.*]] = fir.call @impure() : () -> !fir.heap<!fir.array<10x!fir.logical<4>>>
// CHECK:           %[[VAL_21:.*]] = fir.allocmem !fir.array<?x!fir.logical<4>>, %[[extent:[^ ]*]]
// CHECK:           %[[VAL_22:.*]] = fir.shape %[[extent]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_23:.*]]:2 = hlfir.declare %[[VAL_21]](%{{.*}}) {uniq_name = ".tmp.where"}
// CHECK:           fir.do_loop
// CHECK:             fir.if {{.*}} {
// CHECK:             } else {
// CHECK:               %[[VAL_28:.*]] = hlfir.designate %[[VAL_12]] (%{{.*}})
// CHECK:               %[[VAL_29:.*]] = fir.load %[[VAL_28]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_32:.*]] = hlfir.designate %[[VAL_23]]#0 (%{{.*}})  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               hlfir.assign %[[VAL_29]] to %[[VAL_32]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:             }
// CHECK:           }
// CHECK-NOT:       fir.call @impure
// CHECK:           fir.do_loop
// CHECK:             fir.if {{.*}} {
// CHECK:             } else {
// CHECK:               %[[VAL_42:.*]] = hlfir.designate %[[VAL_23]]#0 (%{{.*}})  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_43:.*]] = fir.load %[[VAL_42]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (!fir.logical<4>) -> i1
// CHECK:               fir.if %[[VAL_44]] {
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-NOT:       fir.call @impure
// CHECK:           fir.do_loop
// CHECK:             fir.if {{.*}} {
// CHECK:             } else {
// CHECK:               %[[VAL_52:.*]] = hlfir.designate %[[VAL_23]]#0 (%{{.*}})  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_53:.*]] = fir.load %[[VAL_52]] : !fir.ref<!fir.logical<4>>
// CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (!fir.logical<4>) -> i1
// CHECK:               fir.if %[[VAL_54]] {
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_21]] : !fir.heap<!fir.array<?x!fir.logical<4>>>
// CHECK:           fir.freemem %[[VAL_12]] : !fir.heap<!fir.array<10x!fir.logical<4>>>
// CHECK:           return
// CHECK:         }
