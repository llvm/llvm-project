// RUN: fir-opt --no-implicit-module --pass-pipeline='func.func(loop-tripcount-versioning)' --verify-each %s | FileCheck %s

// The pass runs on a root func.func with no enclosing module. It previously
// crashed looking the module up for a FirOpBuilder. This is @zero from
// loop-tripcount-versioning.fir; a separate file because --no-implicit-module
// requires a single top-level operation.
func.func @zero(%arg0: !fir.ref<!fir.array<4x26xf64>>, %arg1: !fir.ref<i32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c26 = arith.constant 26 : index
  %c4 = arith.constant 4 : index
  %0 = fir.shape %c4, %c26 : (index, index) -> !fir.shape<2>
  %1 = fir.declare %arg0(%0) uniq_name("tau") : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>) -> !fir.ref<!fir.array<4x26xf64>>
  %2 = fir.load %arg1 : !fir.ref<i32>
  %3 = fir.convert %2 : (i32) -> index
  %4 = arith.maxsi %3, %c0 : index
  %5 = fir.slice %c1, %3, %c1, %c1, %c26, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  fir.do_loop %arg2 = %c1 to %c26 step %c1 unordered {
    fir.do_loop %arg3 = %c1 to %4 step %c1 unordered {
      %6 = fir.array_coor %1(%0) [%5] %arg3, %arg2 : (!fir.ref<!fir.array<4x26xf64>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f64>
      fir.store %cst to %6 : !fir.ref<f64>
    }
  }
  return
}
// CHECK-LABEL: func.func @zero(
// CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
// CHECK:         fir.do_loop
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
// CHECK:           fir.if %[[CMP]] {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step
// CHECK:           } else {
// CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
// CHECK:           }
