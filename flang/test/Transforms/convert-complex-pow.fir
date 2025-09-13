// RUN: fir-opt --convert-complex-pow %s | FileCheck %s

module {
  func.func @pow_c4_i4(%arg0: complex<f32>, %arg1: i32) -> complex<f32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %c1 = fir.convert %arg1 : (i32) -> f32
    %c2 = complex.create %c1, %c0 : complex<f32>
    %0 = complex.pow %arg0, %c2 : complex<f32>
    return %0 : complex<f32>
  }

  func.func @pow_c4_i8(%arg0: complex<f32>, %arg1: i64) -> complex<f32> {
    %c0 = arith.constant 0.000000e+00 : f32
    %c1 = fir.convert %arg1 : (i64) -> f32
    %c2 = complex.create %c1, %c0 : complex<f32>
    %0 = complex.pow %arg0, %c2 : complex<f32>
    return %0 : complex<f32>
  }

  func.func @pow_c4_c4(%arg0: complex<f32>, %arg1: complex<f32>) -> complex<f32> {
    %0 = complex.pow %arg0, %arg1 : complex<f32>
    return %0 : complex<f32>
  }

  func.func @pow_c8_i4(%arg0: complex<f64>, %arg1: i32) -> complex<f64> {
    %c0 = arith.constant 0.000000e+00 : f64
    %c1 = fir.convert %arg1 : (i32) -> f64
    %c2 = complex.create %c1, %c0 : complex<f64>
    %0 = complex.pow %arg0, %c2 : complex<f64>
    return %0 : complex<f64>
  }

  func.func @pow_c8_i8(%arg0: complex<f64>, %arg1: i64) -> complex<f64> {
    %c0 = arith.constant 0.000000e+00 : f64
    %c1 = fir.convert %arg1 : (i64) -> f64
    %c2 = complex.create %c1, %c0 : complex<f64>
    %0 = complex.pow %arg0, %c2 : complex<f64>
    return %0 : complex<f64>
  }

  func.func @pow_c8_c8(%arg0: complex<f64>, %arg1: complex<f64>) -> complex<f64> {
    %0 = complex.pow %arg0, %arg1 : complex<f64>
    return %0 : complex<f64>
  }

  func.func @pow_c16_i4(%arg0: complex<f128>, %arg1: i32) -> complex<f128> {
    %c0 = arith.constant 0.000000e+00 : f128
    %c1 = fir.convert %arg1 : (i32) -> f128
    %c2 = complex.create %c1, %c0 : complex<f128>
    %0 = complex.pow %arg0, %c2 : complex<f128>
    return %0 : complex<f128>
  }

  func.func @pow_c16_i8(%arg0: complex<f128>, %arg1: i64) -> complex<f128> {
    %c0 = arith.constant 0.000000e+00 : f128
    %c1 = fir.convert %arg1 : (i64) -> f128
    %c2 = complex.create %c1, %c0 : complex<f128>
    %0 = complex.pow %arg0, %c2 : complex<f128>
    return %0 : complex<f128>
  }

  func.func @pow_c16_c16(%arg0: complex<f128>, %arg1: complex<f128>) -> complex<f128> {
    %0 = complex.pow %arg0, %arg1 : complex<f128>
    return %0 : complex<f128>
  }
}

// CHECK-LABEL: func.func @pow_c4_i4(
// CHECK: fir.call @_FortranAcpowi(%{{.*}}, %{{.*}}) : (complex<f32>, i32) -> complex<f32>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c4_i8(
// CHECK: fir.call @_FortranAcpowk(%{{.*}}, %{{.*}}) : (complex<f32>, i64) -> complex<f32>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c4_c4(
// CHECK: fir.call @cpowf(%{{.*}}, %{{.*}}) : (complex<f32>, complex<f32>) -> complex<f32>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c8_i4(
// CHECK: fir.call @_FortranAzpowi(%{{.*}}, %{{.*}}) : (complex<f64>, i32) -> complex<f64>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c8_i8(
// CHECK: fir.call @_FortranAzpowk(%{{.*}}, %{{.*}}) : (complex<f64>, i64) -> complex<f64>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c8_c8(
// CHECK: fir.call @cpow(%{{.*}}, %{{.*}}) : (complex<f64>, complex<f64>) -> complex<f64>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c16_i4(
// CHECK: fir.call @_FortranAcqpowi(%{{.*}}, %{{.*}}) : (complex<f128>, i32) -> complex<f128>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c16_i8(
// CHECK: fir.call @_FortranAcqpowk(%{{.*}}, %{{.*}}) : (complex<f128>, i64) -> complex<f128>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c16_c16(
// CHECK: fir.call @_FortranACPowF128(%{{.*}}, %{{.*}}) : (complex<f128>, complex<f128>) -> complex<f128>
// CHECK-NOT: complex.pow
