// Test hlfir.where masked region cleanup lowering (the freemem in the tests).
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

func.func @loop_cleanup(%mask : !fir.ref<!fir.array<2x!fir.logical<4>>>, %x : !fir.ref<!fir.array<2xf32>>, %y : !fir.ref<!fir.array<2xf32>>) {
    hlfir.where {
        %1 = fir.allocmem !fir.array<10xi32>
        hlfir.yield %mask : !fir.ref<!fir.array<2x!fir.logical<4>>> cleanup {
          fir.freemem %1 : !fir.heap<!fir.array<10xi32>>
        }
    } do {
      hlfir.region_assign {
        %1 = fir.allocmem !fir.array<1xi32>
        %2 = fir.allocmem !fir.array<2xi32>
        hlfir.yield %x : !fir.ref<!fir.array<2xf32>> cleanup {
          fir.freemem %2 : !fir.heap<!fir.array<2xi32>>
          fir.freemem %1 : !fir.heap<!fir.array<1xi32>>
        }
      } to {
        %1 = fir.allocmem !fir.array<3xi32>
        %2 = fir.allocmem !fir.array<4xi32>
        hlfir.yield %y : !fir.ref<!fir.array<2xf32>> cleanup {
          fir.freemem %2 : !fir.heap<!fir.array<4xi32>>
          fir.freemem %1 : !fir.heap<!fir.array<3xi32>>
        }
      }
    }
  return
}
// CHECK-LABEL:   func.func @loop_cleanup(
// CHECK:           %[[VAL_3:.*]] = fir.allocmem !fir.array<10xi32>
// CHECK:           fir.do_loop
// CHECK:             fir.if
// CHECK:               %[[VAL_11:.*]] = fir.allocmem !fir.array<1xi32>
// CHECK:               %[[VAL_12:.*]] = fir.allocmem !fir.array<2xi32>
// CHECK:               %[[VAL_14:.*]] = fir.allocmem !fir.array<3xi32>
// CHECK:               %[[VAL_15:.*]] = fir.allocmem !fir.array<4xi32>
// CHECK:               hlfir.assign
// CHECK:               fir.freemem %[[VAL_15]] : !fir.heap<!fir.array<4xi32>>
// CHECK:               fir.freemem %[[VAL_14]] : !fir.heap<!fir.array<3xi32>>
// CHECK:               fir.freemem %[[VAL_12]] : !fir.heap<!fir.array<2xi32>>
// CHECK:               fir.freemem %[[VAL_11]] : !fir.heap<!fir.array<1xi32>>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.freemem %[[VAL_3]] : !fir.heap<!fir.array<10xi32>>
