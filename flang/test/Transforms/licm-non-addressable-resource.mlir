// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Verify that a write of non-addressable AccCurrentDeviceIdResource
// resource does not prevent LICM of fir.load:
// CHECK-LABEL:   func.func @_QPtest(
// CHECK:           %[[DECLARE_1:.*]] = fir.declare %{{.*}} dummy_scope %{{.*}} arg 2 {uniq_name = "_QFtestEy"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<f32>
// CHECK:           fir.do_loop {{.*}} {
// CHECK:             acc.set device_num(%{{.*}} : index)
// CHECK:             fir.store %[[LOAD_0]] to %{{.*}} : !fir.ref<f32>
func.func @_QPtest(%arg0: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.dummy_scope : !fir.dscope
  %3 = fir.shape %c10 : (index) -> !fir.shape<1>
  %4 = fir.declare %arg0(%3) dummy_scope %0 arg 1 {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf32>>
  %5 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtestEy"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  fir.do_loop %arg2 = %c1 to %c10 step %c1 {
    %8 = fir.load %5 : !fir.ref<f32>
    acc.set device_num(%c1 : index)
    %11 = fir.array_coor %4(%3) %arg2 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
    fir.store %8 to %11 : !fir.ref<f32>
  }
  return
}
