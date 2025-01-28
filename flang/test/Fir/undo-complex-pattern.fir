// Test regrouping of + and - operations on complex components into complex operations
// RUN: fir-opt --canonicalize %s | FileCheck %s


// CHECK-LABEL: @add
func.func @add(%z: !fir.ref<complex<f64>>, %z1 : complex<f64>, %z2 : complex<f64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %real1 = fir.extract_value %z1, [0 : index] : (complex<f64>) -> f64
  %imag1 = fir.extract_value %z1, [1 : index] : (complex<f64>) -> f64
  %real2 = fir.extract_value %z2, [0 : index] : (complex<f64>) -> f64
  %imag2 = fir.extract_value %z2, [1 : index] : (complex<f64>) -> f64

  // CHECK-LABEL: fir.addc
  %real = arith.addf %real1, %real2 : f64
  %imag = arith.addf %imag1, %imag2 : f64
  %undef = fir.undefined complex<f64>
  %insert_real = fir.insert_value %undef, %real, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %insert_imag = fir.insert_value %insert_real, %imag, [1 : index] : (complex<f64>, f64) -> complex<f64>
  fir.store %insert_imag to %z : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: @sub
func.func @sub(%z: !fir.ref<complex<f64>>, %z1 : complex<f64>, %z2 : complex<f64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %real1 = fir.extract_value %z1, [0 : index] : (complex<f64>) -> f64
  %imag1 = fir.extract_value %z1, [1 : index] : (complex<f64>) -> f64
  %real2 = fir.extract_value %z2, [0 : index] : (complex<f64>) -> f64
  %imag2 = fir.extract_value %z2, [1 : index] : (complex<f64>) -> f64

  // CHECK-LABEL: fir.subc
  %real = arith.subf %real1, %real2 : f64
  %imag = arith.subf %imag1, %imag2 : f64
  %undef = fir.undefined complex<f64>
  %insert_real = fir.insert_value %undef, %real, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %insert_imag = fir.insert_value %insert_real, %imag, [1 : index] : (complex<f64>, f64) -> complex<f64>
  fir.store %insert_imag to %z : !fir.ref<complex<f64>>
  return
}

// CHECK-LABEL: @undefOpHiddenByBranch
func.func @undefOpHiddenByBranch(%z: !fir.ref<complex<f64>>, %b: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.cond_br %b, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %u1 = fir.undefined complex<f64>
  %z1l = fir.call @bar1() : () -> complex<f64>
  %z1r = fir.call @bar1() : () -> complex<f64>
  cf.br ^bb3(%u1, %z1l, %z1r : complex<f64>, complex<f64>, complex<f64>)
^bb2:  // pred: ^bb0
  %u2 = fir.undefined complex<f64>
  %z2l = fir.call @bar2() : () -> complex<f64>
  %z2r = fir.call @bar2() : () -> complex<f64>
  cf.br ^bb3(%u2, %z2l, %z2r : complex<f64>, complex<f64>, complex<f64>)

// CHECK: ^bb3(%[[z1:.*]]: complex<f64>, %[[z2:.*]]: complex<f64>):  // 2 preds: ^bb1, ^bb2
// CHECK:  fir.addc %[[z1]], %[[z2]] : complex<f64>

^bb3(%undef : complex<f64>, %z1 : complex<f64>, %z2 : complex<f64>):  // 2 preds: ^bb1, ^bb2
  %real1 = fir.extract_value %z1, [0 : index] : (complex<f64>) -> f64
  %imag1 = fir.extract_value %z1, [1 : index] : (complex<f64>) -> f64
  %real2 = fir.extract_value %z2, [0 : index] : (complex<f64>) -> f64
  %imag2 = fir.extract_value %z2, [1 : index] : (complex<f64>) -> f64
  %real = arith.addf %real1, %real2 : f64
  %imag = arith.addf %imag1, %imag2 : f64
  %insert_real = fir.insert_value %undef, %real, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %insert_imag = fir.insert_value %insert_real, %imag, [1 : index] : (complex<f64>, f64) -> complex<f64>
  fir.store %insert_imag to %z : !fir.ref<complex<f64>>
  return
}
func.func private @bar1() -> complex<f64>
func.func private @bar2() -> complex<f64>

// CHECK-LABEL: @close_but_bad_pattern
func.func @close_but_bad_pattern(%z: !fir.ref<complex<f64>>, %z1 : complex<f64>, %z2 : complex<f64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %real1 = fir.extract_value %z1, [0 : index] : (complex<f64>) -> f64
  // extracting %c0 instead of %c1 
  %imag1 = fir.extract_value %z1, [0 : index] : (complex<f64>) -> f64
  %real2 = fir.extract_value %z2, [0 : index] : (complex<f64>) -> f64
  %imag2 = fir.extract_value %z2, [1 : index] : (complex<f64>) -> f64
  // CHECK: arith.subf
  // CHECK: subf
  %real = arith.subf %real1, %real2 : f64
  %imag = arith.subf %imag1, %imag2 : f64
  %undef = fir.undefined complex<f64>
  // CHECK: %[[insert1:.*]] = fir.insert_value %{{.*}}, %{{.*}}, [0
  // CHECK: %[[insert2:.*]] = fir.insert_value %[[insert1]], %{{.*}}, [1
  %insert_real = fir.insert_value %undef, %real, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %insert_imag = fir.insert_value %insert_real, %imag, [1 : index] : (complex<f64>, f64) -> complex<f64>
  // CHECK: fir.store %[[insert2]] to {{.*}}
  fir.store %insert_imag to %z : !fir.ref<complex<f64>>
  return
}
