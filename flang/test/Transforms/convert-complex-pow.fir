// RUN: fir-opt --convert-complex-pow %s | FileCheck %s

module {
  func.func @pow_c4_i4(%arg0: complex<f32>, %arg1: i32) -> complex<f32> {
    %0 = complex.powi %arg0, %arg1 : complex<f32>, i32
    return %0 : complex<f32>
  }

  func.func @pow_c4_i4_fast(%arg0: complex<f32>, %arg1: i32) -> complex<f32> {
    %0 = complex.powi %arg0, %arg1 fastmath<fast> : complex<f32>, i32
    return %0 : complex<f32>
  }

  func.func @pow_c4_i8(%arg0: complex<f32>, %arg1: i64) -> complex<f32> {
    %0 = complex.powi %arg0, %arg1 : complex<f32>, i64
    return %0 : complex<f32>
  }

  func.func @pow_c8_i4(%arg0: complex<f64>, %arg1: i32) -> complex<f64> {
    %0 = complex.powi %arg0, %arg1 : complex<f64>, i32
    return %0 : complex<f64>
  }

  func.func @pow_c8_i8(%arg0: complex<f64>, %arg1: i64) -> complex<f64> {
    %0 = complex.powi %arg0, %arg1 : complex<f64>, i64
    return %0 : complex<f64>
  }

  func.func @pow_c16_i4(%arg0: complex<f128>, %arg1: i32) -> complex<f128> {
    %0 = complex.powi %arg0, %arg1 : complex<f128>, i32
    return %0 : complex<f128>
  }

  func.func @pow_c16_i8(%arg0: complex<f128>, %arg1: i64) -> complex<f128> {
    %0 = complex.powi %arg0, %arg1 : complex<f128>, i64
    return %0 : complex<f128>
  }

  func.func @pow_c4_fast(%arg0: complex<f32>, %arg1: f32) -> complex<f32> {
    %c1 = arith.constant 1.0 : f32
    %0 = complex.create %arg1, %c1 : complex<f32>
    %1 = complex.pow %arg0, %0 fastmath<fast> : complex<f32>
    return %1 : complex<f32>
  }

  func.func @pow_c8_complex(%arg0: complex<f64>, %arg1: f64) -> complex<f64> {
    %c2 = arith.constant 2.0 : f64
    %0 = complex.create %arg1, %c2 : complex<f64>
    %1 = complex.pow %arg0, %0 : complex<f64>
    return %1 : complex<f64>
  }

  func.func @pow_c16_complex(%arg0: complex<f128>, %arg1: f128) -> complex<f128> {
    %c3 = arith.constant 3.0 : f128
    %0 = complex.create %arg1, %c3 : complex<f128>
    %1 = complex.pow %arg0, %0 : complex<f128>
    return %1 : complex<f128>
  }
}

// CHECK-LABEL: func.func @pow_c4_i4(
// CHECK: fir.call @_FortranAcpowi(%{{.*}}, %{{.*}}) : (complex<f32>, i32) -> complex<f32>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c4_i4_fast(
// CHECK: fir.call @_FortranAcpowi(%{{.*}}, %{{.*}}) fastmath<fast> : (complex<f32>, i32) -> complex<f32>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c4_i8(
// CHECK: fir.call @_FortranAcpowk(%{{.*}}, %{{.*}}) : (complex<f32>, i64) -> complex<f32>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c8_i4(
// CHECK: fir.call @_FortranAzpowi(%{{.*}}, %{{.*}}) : (complex<f64>, i32) -> complex<f64>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c8_i8(
// CHECK: fir.call @_FortranAzpowk(%{{.*}}, %{{.*}}) : (complex<f64>, i64) -> complex<f64>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c16_i4(
// CHECK: fir.call @_FortranAcqpowi(%{{.*}}, %{{.*}}) : (complex<f128>, i32) -> complex<f128>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c16_i8(
// CHECK: fir.call @_FortranAcqpowk(%{{.*}}, %{{.*}}) : (complex<f128>, i64) -> complex<f128>
// CHECK-NOT: complex.pow
// CHECK-NOT: complex.powi

// CHECK-LABEL: func.func @pow_c4_fast(
// CHECK: %[[EXP:.*]] = complex.create %{{.*}}, %{{.*}} : complex<f32>
// CHECK: fir.call @cpowf(%{{.*}}, %[[EXP]]) fastmath<fast> : (complex<f32>, complex<f32>) -> complex<f32>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c8_complex(
// CHECK: %[[EXP:.*]] = complex.create %{{.*}}, %{{.*}} : complex<f64>
// CHECK: fir.call @cpow(%{{.*}}, %[[EXP]]) : (complex<f64>, complex<f64>) -> complex<f64>
// CHECK-NOT: complex.pow

// CHECK-LABEL: func.func @pow_c16_complex(
// CHECK: %[[EXP:.*]] = complex.create %{{.*}}, %{{.*}} : complex<f128>
// CHECK: fir.call @_FortranACPowF128(%{{.*}}, %[[EXP]]) : (complex<f128>, complex<f128>) -> complex<f128>
// CHECK-NOT: complex.pow
