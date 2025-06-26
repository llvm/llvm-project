// RUN: fir-opt %s --simplify-hlfir-intrinsics --flang-complex-div-converter | FileCheck %s

func.func @test_double_complex_div(%arg0: !fir.ref<complex<f64>>, %arg1: !fir.ref<complex<f64>>, %arg2: !fir.ref<complex<f64>>) {
  %0 = fir.load %arg1 : !fir.ref<complex<f64>>
  %1 = fir.load %arg2 : !fir.ref<complex<f64>>
  %2 = fir.extract_value %0, [0 : index] : (complex<f64>) -> f64
  %3 = fir.extract_value %0, [1 : index] : (complex<f64>) -> f64
  %4 = fir.extract_value %1, [0 : index] : (complex<f64>) -> f64
  %5 = fir.extract_value %1, [1 : index] : (complex<f64>) -> f64
  %6 = fir.call @__divdc3(%2, %3, %4, %5) fastmath<contract> : (f64, f64, f64, f64) -> complex<f64>
  fir.store %6 to %arg0 : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: func.func @test_double_complex_div
// CHECK-NOT: fir.call @__divdc3
// CHECK: %[[R1:.*]] = arith.mulf %2, %4 : f64
// CHECK: %[[R2:.*]] = arith.mulf %4, %4 : f64
// CHECK: %[[R3:.*]] = arith.mulf %3, %4 : f64
// CHECK: %[[R4:.*]] = arith.mulf %2, %5 : f64
// CHECK: %[[R5:.*]] = arith.mulf %3, %5 : f64
// CHECK: %[[R6:.*]] = arith.mulf %5, %5 : f64
// CHECK: %[[DENOM:.*]] = arith.addf %[[R2]], %[[R6]] : f64
// CHECK: %[[NUM_RE:.*]] = arith.addf %[[R1]], %[[R5]] : f64
// CHECK: %[[NUM_IM:.*]] = arith.subf %[[R3]], %[[R4]] : f64
// CHECK: %[[RES_RE:.*]] = arith.divf %[[NUM_RE]], %[[DENOM]] : f64
// CHECK: %[[RES_IM:.*]] = arith.divf %[[NUM_IM]], %[[DENOM]] : f64
// CHECK: %[[U:.*]] = fir.undefined complex<f64>
// CHECK: %[[C0:.*]] = fir.insert_value %[[U]], %[[RES_RE]], [0 : i32] : (complex<f64>, f64) -> complex<f64>
// CHECK: %[[C1:.*]] = fir.insert_value %[[C0]], %[[RES_IM]], [1 : i32] : (complex<f64>, f64) -> complex<f64>
// CHECK: fir.store %[[C1]] to %arg0 : !fir.ref<complex<f64>>