// Test regrouping of + and - operations on complex components into complex operations
// RUN: fir-opt --canonicalize %s | FileCheck %s


// CHECK-LABEL: @add
func @add(%z: !fir.ref<!fir.complex<8>>, %z1 : !fir.complex<8>, %z2 : !fir.complex<8>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %real1 = fir.extract_value %z1, %c0 : (!fir.complex<8>, index) -> f64
  %imag1 = fir.extract_value %z1, %c1 : (!fir.complex<8>, index) -> f64
  %real2 = fir.extract_value %z2, %c0 : (!fir.complex<8>, index) -> f64
  %imag2 = fir.extract_value %z2, %c1 : (!fir.complex<8>, index) -> f64

  // CHECK-LABEL: fir.addc
  %real = addf %real1, %real2 : f64
  %imag = addf %imag1, %imag2 : f64
  %undef = fir.undefined !fir.complex<8>
  %insert_real = fir.insert_value %undef, %real, %c0 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  %insert_imag = fir.insert_value %insert_real, %imag, %c1 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  fir.store %insert_imag to %z : !fir.ref<!fir.complex<8>>
  return
}

// CHECK-LABEL: @sub
func @sub(%z: !fir.ref<!fir.complex<8>>, %z1 : !fir.complex<8>, %z2 : !fir.complex<8>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %real1 = fir.extract_value %z1, %c0 : (!fir.complex<8>, index) -> f64
  %imag1 = fir.extract_value %z1, %c1 : (!fir.complex<8>, index) -> f64
  %real2 = fir.extract_value %z2, %c0 : (!fir.complex<8>, index) -> f64
  %imag2 = fir.extract_value %z2, %c1 : (!fir.complex<8>, index) -> f64

  // CHECK-LABEL: fir.subc
  %real = subf %real1, %real2 : f64
  %imag = subf %imag1, %imag2 : f64
  %undef = fir.undefined !fir.complex<8>
  %insert_real = fir.insert_value %undef, %real, %c0 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  %insert_imag = fir.insert_value %insert_real, %imag, %c1 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  fir.store %insert_imag to %z : !fir.ref<!fir.complex<8>>
  return
}

// CHECK-LABEL: @undefOpHiddenByBranch
func @undefOpHiddenByBranch(%z: !fir.ref<!fir.complex<8>>, %b: i1) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  cond_br %b, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %u1 = fir.undefined !fir.complex<8>
  %z1l = fir.call @bar1() : () -> !fir.complex<8>
  %z1r = fir.call @bar1() : () -> !fir.complex<8>
  br ^bb3(%u1, %z1l, %z1r : !fir.complex<8>, !fir.complex<8>, !fir.complex<8>)
^bb2:  // pred: ^bb0
  %u2 = fir.undefined !fir.complex<8>
  %z2l = fir.call @bar2() : () -> !fir.complex<8>
  %z2r = fir.call @bar2() : () -> !fir.complex<8>
  br ^bb3(%u2, %z2l, %z2r : !fir.complex<8>, !fir.complex<8>, !fir.complex<8>)

// CHECK: ^bb3(%[[z1:.*]]: !fir.complex<8>, %[[z2:.*]]: !fir.complex<8>):  // 2 preds: ^bb1, ^bb2
// CHECK:  fir.addc %[[z1]], %[[z2]] : !fir.complex<8>

^bb3(%undef : !fir.complex<8>, %z1 : !fir.complex<8>, %z2 : !fir.complex<8>):  // 2 preds: ^bb1, ^bb2
  %real1 = fir.extract_value %z1, %c0 : (!fir.complex<8>, index) -> f64
  %imag1 = fir.extract_value %z1, %c1 : (!fir.complex<8>, index) -> f64
  %real2 = fir.extract_value %z2, %c0 : (!fir.complex<8>, index) -> f64
  %imag2 = fir.extract_value %z2, %c1 : (!fir.complex<8>, index) -> f64
  %real = addf %real1, %real2 : f64
  %imag = addf %imag1, %imag2 : f64
  %insert_real = fir.insert_value %undef, %real, %c0 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  %insert_imag = fir.insert_value %insert_real, %imag, %c1 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  fir.store %insert_imag to %z : !fir.ref<!fir.complex<8>>
  return
}
func private @bar1() -> !fir.complex<8>
func private @bar2() -> !fir.complex<8>

// CHECK-LABEL: @close_but_bad_pattern
func @close_but_bad_pattern(%z: !fir.ref<!fir.complex<8>>, %z1 : !fir.complex<8>, %z2 : !fir.complex<8>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %real1 = fir.extract_value %z1, %c0 : (!fir.complex<8>, index) -> f64
  // extracting %c0 instead of %c1 
  %imag1 = fir.extract_value %z1, %c0 : (!fir.complex<8>, index) -> f64
  %real2 = fir.extract_value %z2, %c0 : (!fir.complex<8>, index) -> f64
  %imag2 = fir.extract_value %z2, %c1 : (!fir.complex<8>, index) -> f64
  // CHECK: subf
  // CHECK: subf
  %real = subf %real1, %real2 : f64
  %imag = subf %imag1, %imag2 : f64
  %undef = fir.undefined !fir.complex<8>
  // CHECK: %[[insert1:.*]] = fir.insert_value %{{.*}}, %{{.*}}, %{{.*}}
  // CHECK: %[[insert2:.*]] = fir.insert_value %[[insert1]], %{{.*}}, %{{.*}}
  %insert_real = fir.insert_value %undef, %real, %c0 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  %insert_imag = fir.insert_value %insert_real, %imag, %c1 : (!fir.complex<8>, f64, index) -> !fir.complex<8>
  // CHECK: fir.store %[[insert2]] to {{.*}}
  fir.store %insert_imag to %z : !fir.ref<!fir.complex<8>>
  return
}
