// RUN: split-file %s %t
// TODO: verify that Fast-Math Flags and 'strictfp' are properly set.

//--- abs_fast.fir
// RUN: fir-opt %t/abs_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/abs_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.fabs({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.fabs({{%[A-Za-z0-9._]+}}) : (f64) -> f64

// CHECK: @_QPtest_real16
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.fabs({{%[A-Za-z0-9._]+}}) : (f128) -> f128

// CHECK: @_QPtest_complex4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @hypotf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_complex8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @hypot({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.absf %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.absf %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func @_QPtest_real16(%arg0: !fir.ref<f128> {fir.bindc_name = "x"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "test_real16", uniq_name = "_QFtest_real16Etest_real16"}
  %1 = fir.load %arg0 : !fir.ref<f128>
  %2 = math.absf %1 : f128
  fir.store %2 to %0 : !fir.ref<f128>
  %3 = fir.load %0 : !fir.ref<f128>
  return %3 : f128
}
func.func @_QPtest_complex4(%arg0: !fir.ref<complex<f32>> {fir.bindc_name = "c"}) -> complex<f32> {
  %0 = fir.alloca complex<f32> {bindc_name = "test_complex4", uniq_name = "_QFtest_complex4Etest_complex4"}
  %1 = fir.load %arg0 : !fir.ref<complex<f32>>
  %2 = fir.extract_value %1, [0 : index] : (complex<f32>) -> f32
  %3 = fir.extract_value %1, [1 : index] : (complex<f32>) -> f32
  %4 = fir.call @hypotf(%2, %3) : (f32, f32) -> f32
  %cst = arith.constant 0.000000e+00 : f32
  %5 = fir.undefined complex<f32>
  %6 = fir.insert_value %5, %4, [0 : index] : (complex<f32>, f32) -> complex<f32>
  %7 = fir.insert_value %6, %cst, [1 : index] : (complex<f32>, f32) -> complex<f32>
  fir.store %7 to %0 : !fir.ref<complex<f32>>
  %8 = fir.load %0 : !fir.ref<complex<f32>>
  return %8 : complex<f32>
}
func.func @_QPtest_complex8(%arg0: !fir.ref<complex<f64>> {fir.bindc_name = "c"}) -> complex<f64> {
  %0 = fir.alloca complex<f64> {bindc_name = "test_complex8", uniq_name = "_QFtest_complex8Etest_complex8"}
  %1 = fir.load %arg0 : !fir.ref<complex<f64>>
  %2 = fir.extract_value %1, [0 : index] : (complex<f64>) -> f64
  %3 = fir.extract_value %1, [1 : index] : (complex<f64>) -> f64
  %4 = fir.call @hypot(%2, %3) : (f64, f64) -> f64
  %cst = arith.constant 0.000000e+00 : f64
  %5 = fir.undefined complex<f64>
  %6 = fir.insert_value %5, %4, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %7 = fir.insert_value %6, %cst, [1 : index] : (complex<f64>, f64) -> complex<f64>
  fir.store %7 to %0 : !fir.ref<complex<f64>>
  %8 = fir.load %0 : !fir.ref<complex<f64>>
  return %8 : complex<f64>
}
func.func private @hypotf(f32, f32) -> f32
func.func private @hypot(f64, f64) -> f64

//--- abs_relaxed.fir
// RUN: fir-opt %t/abs_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/abs_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.fabs({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.fabs({{%[A-Za-z0-9._]+}}) : (f64) -> f64

// CHECK: @_QPtest_real16
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.fabs({{%[A-Za-z0-9._]+}}) : (f128) -> f128

// CHECK: @_QPtest_complex4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @hypotf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_complex8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @hypot({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.absf %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.absf %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func @_QPtest_real16(%arg0: !fir.ref<f128> {fir.bindc_name = "x"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "test_real16", uniq_name = "_QFtest_real16Etest_real16"}
  %1 = fir.load %arg0 : !fir.ref<f128>
  %2 = math.absf %1 : f128
  fir.store %2 to %0 : !fir.ref<f128>
  %3 = fir.load %0 : !fir.ref<f128>
  return %3 : f128
}
func.func @_QPtest_complex4(%arg0: !fir.ref<complex<f32>> {fir.bindc_name = "c"}) -> complex<f32> {
  %0 = fir.alloca complex<f32> {bindc_name = "test_complex4", uniq_name = "_QFtest_complex4Etest_complex4"}
  %1 = fir.load %arg0 : !fir.ref<complex<f32>>
  %2 = fir.extract_value %1, [0 : index] : (complex<f32>) -> f32
  %3 = fir.extract_value %1, [1 : index] : (complex<f32>) -> f32
  %4 = fir.call @hypotf(%2, %3) : (f32, f32) -> f32
  %cst = arith.constant 0.000000e+00 : f32
  %5 = fir.undefined complex<f32>
  %6 = fir.insert_value %5, %4, [0 : index] : (complex<f32>, f32) -> complex<f32>
  %7 = fir.insert_value %6, %cst, [1 : index] : (complex<f32>, f32) -> complex<f32>
  fir.store %7 to %0 : !fir.ref<complex<f32>>
  %8 = fir.load %0 : !fir.ref<complex<f32>>
  return %8 : complex<f32>
}
func.func @_QPtest_complex8(%arg0: !fir.ref<complex<f64>> {fir.bindc_name = "c"}) -> complex<f64> {
  %0 = fir.alloca complex<f64> {bindc_name = "test_complex8", uniq_name = "_QFtest_complex8Etest_complex8"}
  %1 = fir.load %arg0 : !fir.ref<complex<f64>>
  %2 = fir.extract_value %1, [0 : index] : (complex<f64>) -> f64
  %3 = fir.extract_value %1, [1 : index] : (complex<f64>) -> f64
  %4 = fir.call @hypot(%2, %3) : (f64, f64) -> f64
  %cst = arith.constant 0.000000e+00 : f64
  %5 = fir.undefined complex<f64>
  %6 = fir.insert_value %5, %4, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %7 = fir.insert_value %6, %cst, [1 : index] : (complex<f64>, f64) -> complex<f64>
  fir.store %7 to %0 : !fir.ref<complex<f64>>
  %8 = fir.load %0 : !fir.ref<complex<f64>>
  return %8 : complex<f64>
}
func.func private @hypotf(f32, f32) -> f32
func.func private @hypot(f64, f64) -> f64

//--- abs_precise.fir
// RUN: fir-opt %t/abs_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/abs_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @fabsf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @fabs({{%[A-Za-z0-9._]+}}) : (f64) -> f64

// CHECK: @_QPtest_real16
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.fabs.f128({{%[A-Za-z0-9._]+}}) : (f128) -> f128

// CHECK: @_QPtest_complex4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @hypotf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_complex8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @hypot({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @fabsf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @fabs(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func @_QPtest_real16(%arg0: !fir.ref<f128> {fir.bindc_name = "x"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "test_real16", uniq_name = "_QFtest_real16Etest_real16"}
  %1 = fir.load %arg0 : !fir.ref<f128>
  %2 = fir.call @llvm.fabs.f128(%1) : (f128) -> f128
  fir.store %2 to %0 : !fir.ref<f128>
  %3 = fir.load %0 : !fir.ref<f128>
  return %3 : f128
}
func.func @_QPtest_complex4(%arg0: !fir.ref<complex<f32>> {fir.bindc_name = "c"}) -> complex<f32> {
  %0 = fir.alloca complex<f32> {bindc_name = "test_complex4", uniq_name = "_QFtest_complex4Etest_complex4"}
  %1 = fir.load %arg0 : !fir.ref<complex<f32>>
  %2 = fir.extract_value %1, [0 : index] : (complex<f32>) -> f32
  %3 = fir.extract_value %1, [1 : index] : (complex<f32>) -> f32
  %4 = fir.call @hypotf(%2, %3) : (f32, f32) -> f32
  %cst = arith.constant 0.000000e+00 : f32
  %5 = fir.undefined complex<f32>
  %6 = fir.insert_value %5, %4, [0 : index] : (complex<f32>, f32) -> complex<f32>
  %7 = fir.insert_value %6, %cst, [1 : index] : (complex<f32>, f32) -> complex<f32>
  fir.store %7 to %0 : !fir.ref<complex<f32>>
  %8 = fir.load %0 : !fir.ref<complex<f32>>
  return %8 : complex<f32>
}
func.func @_QPtest_complex8(%arg0: !fir.ref<complex<f64>> {fir.bindc_name = "c"}) -> complex<f64> {
  %0 = fir.alloca complex<f64> {bindc_name = "test_complex8", uniq_name = "_QFtest_complex8Etest_complex8"}
  %1 = fir.load %arg0 : !fir.ref<complex<f64>>
  %2 = fir.extract_value %1, [0 : index] : (complex<f64>) -> f64
  %3 = fir.extract_value %1, [1 : index] : (complex<f64>) -> f64
  %4 = fir.call @hypot(%2, %3) : (f64, f64) -> f64
  %cst = arith.constant 0.000000e+00 : f64
  %5 = fir.undefined complex<f64>
  %6 = fir.insert_value %5, %4, [0 : index] : (complex<f64>, f64) -> complex<f64>
  %7 = fir.insert_value %6, %cst, [1 : index] : (complex<f64>, f64) -> complex<f64>
  fir.store %7 to %0 : !fir.ref<complex<f64>>
  %8 = fir.load %0 : !fir.ref<complex<f64>>
  return %8 : complex<f64>
}
func.func private @fabsf(f32) -> f32
func.func private @fabs(f64) -> f64
func.func private @llvm.fabs.f128(f128) -> f128
func.func private @hypotf(f32, f32) -> f32
func.func private @hypot(f64, f64) -> f64

//--- aint_fast.fir
// RUN: fir-opt %t/aint_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/aint_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.trunc.f32({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.trunc.f64({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.trunc.f32(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.trunc.f64(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @llvm.trunc.f32(f32) -> f32
func.func private @llvm.trunc.f64(f64) -> f64

//--- aint_relaxed.fir
// RUN: fir-opt %t/aint_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/aint_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.trunc.f32({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.trunc.f64({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.trunc.f32(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.trunc.f64(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @llvm.trunc.f32(f32) -> f32
func.func private @llvm.trunc.f64(f64) -> f64

//--- aint_precise.fir
// RUN: fir-opt %t/aint_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/aint_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.trunc.f32({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.trunc.f64({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.trunc.f32(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.trunc.f64(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @llvm.trunc.f32(f32) -> f32
func.func private @llvm.trunc.f64(f64) -> f64

//--- anint_fast.fir
// RUN: fir-opt %t/anint_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/anint_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = llvm.intr.round(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = llvm.intr.round(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- anint_relaxed.fir
// RUN: fir-opt %t/anint_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/anint_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.round({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = llvm.intr.round(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = llvm.intr.round(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- anint_precise.fir
// RUN: fir-opt %t/anint_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/anint_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.round.f32({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.round.f64({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.round.f32(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.round.f64(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @llvm.round.f32(f32) -> f32
func.func private @llvm.round.f64(f64) -> f64

//--- atan_fast.fir
// RUN: fir-opt %t/atan_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/atan_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.atan %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.atan %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- atan_relaxed.fir
// RUN: fir-opt %t/atan_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/atan_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.atan %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.atan %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- atan_precise.fir
// RUN: fir-opt %t/atan_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/atan_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @atanf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @atan(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @atanf(f32) -> f32
func.func private @atan(f64) -> f64

//--- atan2_fast.fir
// RUN: fir-opt %t/atan2_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/atan2_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan2f({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan2({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg1 : !fir.ref<f32>
  %3 = math.atan2 %1, %2 : f32
  fir.store %3 to %0 : !fir.ref<f32>
  %4 = fir.load %0 : !fir.ref<f32>
  return %4 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg1 : !fir.ref<f64>
  %3 = math.atan2 %1, %2 : f64
  fir.store %3 to %0 : !fir.ref<f64>
  %4 = fir.load %0 : !fir.ref<f64>
  return %4 : f64
}

//--- atan2_relaxed.fir
// RUN: fir-opt %t/atan2_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/atan2_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan2f({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan2({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg1 : !fir.ref<f32>
  %3 = math.atan2 %1, %2 : f32
  fir.store %3 to %0 : !fir.ref<f32>
  %4 = fir.load %0 : !fir.ref<f32>
  return %4 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg1 : !fir.ref<f64>
  %3 = math.atan2 %1, %2 : f64
  fir.store %3 to %0 : !fir.ref<f64>
  %4 = fir.load %0 : !fir.ref<f64>
  return %4 : f64
}

//--- atan2_precise.fir
// RUN: fir-opt %t/atan2_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/atan2_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan2f({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @atan2({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg1 : !fir.ref<f32>
  %3 = fir.call @atan2f(%1, %2) : (f32, f32) -> f32
  fir.store %3 to %0 : !fir.ref<f32>
  %4 = fir.load %0 : !fir.ref<f32>
  return %4 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg1 : !fir.ref<f64>
  %3 = fir.call @atan2(%1, %2) : (f64, f64) -> f64
  fir.store %3 to %0 : !fir.ref<f64>
  %4 = fir.load %0 : !fir.ref<f64>
  return %4 : f64
}
func.func private @atan2f(f32, f32) -> f32
func.func private @atan2(f64, f64) -> f64

//--- ceiling_fast.fir
// RUN: fir-opt %t/ceiling_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/ceiling_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.ceil({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.ceil({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.ceil %1 : f32
  %3 = fir.convert %2 : (f32) -> i32
  %4 = fir.convert %3 : (i32) -> f32
  fir.store %4 to %0 : !fir.ref<f32>
  %5 = fir.load %0 : !fir.ref<f32>
  return %5 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.ceil %1 : f64
  %3 = fir.convert %2 : (f64) -> i32
  %4 = fir.convert %3 : (i32) -> f64
  fir.store %4 to %0 : !fir.ref<f64>
  %5 = fir.load %0 : !fir.ref<f64>
  return %5 : f64
}

//--- ceiling_relaxed.fir
// RUN: fir-opt %t/ceiling_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/ceiling_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.ceil({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.ceil({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.ceil %1 : f32
  %3 = fir.convert %2 : (f32) -> i32
  %4 = fir.convert %3 : (i32) -> f32
  fir.store %4 to %0 : !fir.ref<f32>
  %5 = fir.load %0 : !fir.ref<f32>
  return %5 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.ceil %1 : f64
  %3 = fir.convert %2 : (f64) -> i32
  %4 = fir.convert %3 : (i32) -> f64
  fir.store %4 to %0 : !fir.ref<f64>
  %5 = fir.load %0 : !fir.ref<f64>
  return %5 : f64
}

//--- ceiling_precise.fir
// RUN: fir-opt %t/ceiling_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/ceiling_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @ceilf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @ceil({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @ceilf(%1) : (f32) -> f32
  %3 = fir.convert %2 : (f32) -> i32
  %4 = fir.convert %3 : (i32) -> f32
  fir.store %4 to %0 : !fir.ref<f32>
  %5 = fir.load %0 : !fir.ref<f32>
  return %5 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @ceil(%1) : (f64) -> f64
  %3 = fir.convert %2 : (f64) -> i32
  %4 = fir.convert %3 : (i32) -> f64
  fir.store %4 to %0 : !fir.ref<f64>
  %5 = fir.load %0 : !fir.ref<f64>
  return %5 : f64
}
func.func private @ceilf(f32) -> f32
func.func private @ceil(f64) -> f64

//--- cos_fast.fir
// RUN: fir-opt %t/cos_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/cos_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.cos({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.cos({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.cos %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.cos %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- cos_relaxed.fir
// RUN: fir-opt %t/cos_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/cos_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.cos({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.cos({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.cos %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.cos %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- cos_precise.fir
// RUN: fir-opt %t/cos_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/cos_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @cosf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @cos({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @cosf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @cos(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @cosf(f32) -> f32
func.func private @cos(f64) -> f64

//--- cosh_fast.fir
// RUN: fir-opt %t/cosh_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/cosh_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @coshf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @cosh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @coshf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @cosh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @coshf(f32) -> f32
func.func private @cosh(f64) -> f64

//--- cosh_relaxed.fir
// RUN: fir-opt %t/cosh_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/cosh_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @coshf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @cosh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @coshf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @cosh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @coshf(f32) -> f32
func.func private @cosh(f64) -> f64

//--- cosh_precise.fir
// RUN: fir-opt %t/cosh_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/cosh_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @coshf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @cosh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @coshf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @cosh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @coshf(f32) -> f32
func.func private @cosh(f64) -> f64

//--- erf_fast.fir
// RUN: fir-opt %t/erf_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/erf_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @erff({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @erf({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.erf %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.erf %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- erf_relaxed.fir
// RUN: fir-opt %t/erf_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/erf_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @erff({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @erf({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.erf %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.erf %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- erf_precise.fir
// RUN: fir-opt %t/erf_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/erf_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @erff({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @erf({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @erff(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @erf(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @erff(f32) -> f32
func.func private @erf(f64) -> f64

//--- exp_fast.fir
// RUN: fir-opt %t/exp_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exp_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.exp({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.exp({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.exp %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.exp %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- exp_relaxed.fir
// RUN: fir-opt %t/exp_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exp_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.exp({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.exp({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.exp %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.exp %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- exp_precise.fir
// RUN: fir-opt %t/exp_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exp_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @expf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @exp({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @expf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @exp(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @expf(f32) -> f32
func.func private @exp(f64) -> f64

//--- floor_fast.fir
// RUN: fir-opt %t/floor_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/floor_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.floor({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.floor({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.floor %1 : f32
  %3 = fir.convert %2 : (f32) -> i32
  %4 = fir.convert %3 : (i32) -> f32
  fir.store %4 to %0 : !fir.ref<f32>
  %5 = fir.load %0 : !fir.ref<f32>
  return %5 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.floor %1 : f64
  %3 = fir.convert %2 : (f64) -> i32
  %4 = fir.convert %3 : (i32) -> f64
  fir.store %4 to %0 : !fir.ref<f64>
  %5 = fir.load %0 : !fir.ref<f64>
  return %5 : f64
}

//--- floor_relaxed.fir
// RUN: fir-opt %t/floor_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/floor_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.floor({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.floor({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.floor %1 : f32
  %3 = fir.convert %2 : (f32) -> i32
  %4 = fir.convert %3 : (i32) -> f32
  fir.store %4 to %0 : !fir.ref<f32>
  %5 = fir.load %0 : !fir.ref<f32>
  return %5 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.floor %1 : f64
  %3 = fir.convert %2 : (f64) -> i32
  %4 = fir.convert %3 : (i32) -> f64
  fir.store %4 to %0 : !fir.ref<f64>
  %5 = fir.load %0 : !fir.ref<f64>
  return %5 : f64
}

//--- floor_precise.fir
// RUN: fir-opt %t/floor_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/floor_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @floorf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @floor({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @floorf(%1) : (f32) -> f32
  %3 = fir.convert %2 : (f32) -> i32
  %4 = fir.convert %3 : (i32) -> f32
  fir.store %4 to %0 : !fir.ref<f32>
  %5 = fir.load %0 : !fir.ref<f32>
  return %5 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @floor(%1) : (f64) -> f64
  %3 = fir.convert %2 : (f64) -> i32
  %4 = fir.convert %3 : (i32) -> f64
  fir.store %4 to %0 : !fir.ref<f64>
  %5 = fir.load %0 : !fir.ref<f64>
  return %5 : f64
}
func.func private @floorf(f32) -> f32
func.func private @floor(f64) -> f64

//--- log_fast.fir
// RUN: fir-opt %t/log_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/log_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.log %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.log %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- log_relaxed.fir
// RUN: fir-opt %t/log_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/log_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.log %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.log %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- log_precise.fir
// RUN: fir-opt %t/log_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/log_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @logf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @log({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @logf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @log(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @logf(f32) -> f32
func.func private @log(f64) -> f64

//--- log10_fast.fir
// RUN: fir-opt %t/log10_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/log10_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log10({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log10({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.log10 %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.log10 %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- log10_relaxed.fir
// RUN: fir-opt %t/log10_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/log10_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log10({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.log10({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.log10 %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.log10 %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- log10_precise.fir
// RUN: fir-opt %t/log10_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/log10_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @log10f({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @log10({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @log10f(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @log10(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @log10f(f32) -> f32
func.func private @log10(f64) -> f64

//--- nint_fast.fir
// RUN: fir-opt %t/nint_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/nint_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i32.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i64.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i64

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i32.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i64.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.lround.i32.f32(%1) : (f32) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.load %arg0 : !fir.ref<f32>
  %5 = fir.call @llvm.lround.i64.f32(%4) : (f32) -> i64
  %6 = arith.addi %3, %5 : i64
  %7 = fir.convert %6 : (i64) -> f32
  fir.store %7 to %0 : !fir.ref<f32>
  %8 = fir.load %0 : !fir.ref<f32>
  return %8 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.lround.i32.f64(%1) : (f64) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.load %arg0 : !fir.ref<f64>
  %5 = fir.call @llvm.lround.i64.f64(%4) : (f64) -> i64
  %6 = arith.addi %3, %5 : i64
  %7 = fir.convert %6 : (i64) -> f64
  fir.store %7 to %0 : !fir.ref<f64>
  %8 = fir.load %0 : !fir.ref<f64>
  return %8 : f64
}
func.func private @llvm.lround.i32.f32(f32) -> i32
func.func private @llvm.lround.i64.f32(f32) -> i64
func.func private @llvm.lround.i32.f64(f64) -> i32
func.func private @llvm.lround.i64.f64(f64) -> i64

//--- nint_relaxed.fir
// RUN: fir-opt %t/nint_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/nint_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i32.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i64.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i64

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i32.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i64.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.lround.i32.f32(%1) : (f32) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.load %arg0 : !fir.ref<f32>
  %5 = fir.call @llvm.lround.i64.f32(%4) : (f32) -> i64
  %6 = arith.addi %3, %5 : i64
  %7 = fir.convert %6 : (i64) -> f32
  fir.store %7 to %0 : !fir.ref<f32>
  %8 = fir.load %0 : !fir.ref<f32>
  return %8 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.lround.i32.f64(%1) : (f64) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.load %arg0 : !fir.ref<f64>
  %5 = fir.call @llvm.lround.i64.f64(%4) : (f64) -> i64
  %6 = arith.addi %3, %5 : i64
  %7 = fir.convert %6 : (i64) -> f64
  fir.store %7 to %0 : !fir.ref<f64>
  %8 = fir.load %0 : !fir.ref<f64>
  return %8 : f64
}
func.func private @llvm.lround.i32.f32(f32) -> i32
func.func private @llvm.lround.i64.f32(f32) -> i64
func.func private @llvm.lround.i32.f64(f64) -> i32
func.func private @llvm.lround.i64.f64(f64) -> i64

//--- nint_precise.fir
// RUN: fir-opt %t/nint_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/nint_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i32.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i64.f32({{%[A-Za-z0-9._]+}}) : (f32) -> i64

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i32.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.lround.i64.f64({{%[A-Za-z0-9._]+}}) : (f64) -> i64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @llvm.lround.i32.f32(%1) : (f32) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.load %arg0 : !fir.ref<f32>
  %5 = fir.call @llvm.lround.i64.f32(%4) : (f32) -> i64
  %6 = arith.addi %3, %5 : i64
  %7 = fir.convert %6 : (i64) -> f32
  fir.store %7 to %0 : !fir.ref<f32>
  %8 = fir.load %0 : !fir.ref<f32>
  return %8 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @llvm.lround.i32.f64(%1) : (f64) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.load %arg0 : !fir.ref<f64>
  %5 = fir.call @llvm.lround.i64.f64(%4) : (f64) -> i64
  %6 = arith.addi %3, %5 : i64
  %7 = fir.convert %6 : (i64) -> f64
  fir.store %7 to %0 : !fir.ref<f64>
  %8 = fir.load %0 : !fir.ref<f64>
  return %8 : f64
}
func.func private @llvm.lround.i32.f32(f32) -> i32
func.func private @llvm.lround.i64.f32(f32) -> i64
func.func private @llvm.lround.i32.f64(f64) -> i32
func.func private @llvm.lround.i64.f64(f64) -> i64

//--- exponentiation_fast.fir
// RUN: fir-opt %t/exponentiation_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exponentiation_fast.fir
// CHECK: @_QPtest_real4
// CHECK: [[STOI:%[A-Za-z0-9._]+]] = llvm.sext {{%[A-Za-z0-9._]+}} : i16 to i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f32, i32) -> f32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, i32) -> f32

// CHECK: @_QPtest_real8
// CHECK: [[STOI:%[A-Za-z0-9._]+]] = llvm.sext {{%[A-Za-z0-9._]+}} : i16 to i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f64, i32) -> f64
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, i32) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}, %arg2: !fir.ref<i16> {fir.bindc_name = "s"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg2 : !fir.ref<i16>
  %3 = fir.convert %2 : (i16) -> i32
  %4 = fir.call @llvm.powi.f32.i32(%1, %3) : (f32, i32) -> f32
  %5 = fir.load %arg0 : !fir.ref<f32>
  %6 = fir.load %arg1 : !fir.ref<f32>
  %7 = math.powf %5, %6 : f32
  %8 = arith.addf %4, %7 : f32
  %9 = fir.load %arg0 : !fir.ref<f32>
  %10 = fir.load %arg3 : !fir.ref<i32>
  %11 = fir.call @llvm.powi.f32.i32(%9, %10) : (f32, i32) -> f32
  %12 = arith.addf %8, %11 : f32
  fir.store %12 to %0 : !fir.ref<f32>
  %13 = fir.load %0 : !fir.ref<f32>
  return %13 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}, %arg2: !fir.ref<i16> {fir.bindc_name = "s"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg2 : !fir.ref<i16>
  %3 = fir.convert %2 : (i16) -> i32
  %4 = fir.call @llvm.powi.f64.i32(%1, %3) : (f64, i32) -> f64
  %5 = fir.load %arg0 : !fir.ref<f64>
  %6 = fir.load %arg1 : !fir.ref<f64>
  %7 = math.powf %5, %6 : f64
  %8 = arith.addf %4, %7 : f64
  %9 = fir.load %arg0 : !fir.ref<f64>
  %10 = fir.load %arg3 : !fir.ref<i32>
  %11 = fir.call @llvm.powi.f64.i32(%9, %10) : (f64, i32) -> f64
  %12 = arith.addf %8, %11 : f64
  fir.store %12 to %0 : !fir.ref<f64>
  %13 = fir.load %0 : !fir.ref<f64>
  return %13 : f64
}
func.func private @llvm.powi.f32.i32(f32, i32) -> f32
func.func private @llvm.powi.f64.i32(f64, i32) -> f64

//--- exponentiation_relaxed.fir
// RUN: fir-opt %t/exponentiation_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exponentiation_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: [[STOI:%[A-Za-z0-9._]+]] = llvm.sext {{%[A-Za-z0-9._]+}} : i16 to i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f32, i32) -> f32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, i32) -> f32

// CHECK: @_QPtest_real8
// CHECK: [[STOI:%[A-Za-z0-9._]+]] = llvm.sext {{%[A-Za-z0-9._]+}} : i16 to i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f64, i32) -> f64
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, i32) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}, %arg2: !fir.ref<i16> {fir.bindc_name = "s"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg2 : !fir.ref<i16>
  %3 = fir.convert %2 : (i16) -> i32
  %4 = fir.call @llvm.powi.f32.i32(%1, %3) : (f32, i32) -> f32
  %5 = fir.load %arg0 : !fir.ref<f32>
  %6 = fir.load %arg1 : !fir.ref<f32>
  %7 = math.powf %5, %6 : f32
  %8 = arith.addf %4, %7 : f32
  %9 = fir.load %arg0 : !fir.ref<f32>
  %10 = fir.load %arg3 : !fir.ref<i32>
  %11 = fir.call @llvm.powi.f32.i32(%9, %10) : (f32, i32) -> f32
  %12 = arith.addf %8, %11 : f32
  fir.store %12 to %0 : !fir.ref<f32>
  %13 = fir.load %0 : !fir.ref<f32>
  return %13 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}, %arg2: !fir.ref<i16> {fir.bindc_name = "s"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg2 : !fir.ref<i16>
  %3 = fir.convert %2 : (i16) -> i32
  %4 = fir.call @llvm.powi.f64.i32(%1, %3) : (f64, i32) -> f64
  %5 = fir.load %arg0 : !fir.ref<f64>
  %6 = fir.load %arg1 : !fir.ref<f64>
  %7 = math.powf %5, %6 : f64
  %8 = arith.addf %4, %7 : f64
  %9 = fir.load %arg0 : !fir.ref<f64>
  %10 = fir.load %arg3 : !fir.ref<i32>
  %11 = fir.call @llvm.powi.f64.i32(%9, %10) : (f64, i32) -> f64
  %12 = arith.addf %8, %11 : f64
  fir.store %12 to %0 : !fir.ref<f64>
  %13 = fir.load %0 : !fir.ref<f64>
  return %13 : f64
}
func.func private @llvm.powi.f32.i32(f32, i32) -> f32
func.func private @llvm.powi.f64.i32(f64, i32) -> f64

//--- exponentiation_precise.fir
// RUN: fir-opt %t/exponentiation_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exponentiation_precise.fir
// CHECK: @_QPtest_real4
// CHECK: [[STOI:%[A-Za-z0-9._]+]] = llvm.sext {{%[A-Za-z0-9._]+}} : i16 to i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f32, i32) -> f32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @powf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f32.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, i32) -> f32

// CHECK: @_QPtest_real8
// CHECK: [[STOI:%[A-Za-z0-9._]+]] = llvm.sext {{%[A-Za-z0-9._]+}} : i16 to i32
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, [[STOI]]) : (f64, i32) -> f64
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @pow({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.powi.f64.i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, i32) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}, %arg2: !fir.ref<i16> {fir.bindc_name = "s"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg2 : !fir.ref<i16>
  %3 = fir.convert %2 : (i16) -> i32
  %4 = fir.call @llvm.powi.f32.i32(%1, %3) : (f32, i32) -> f32
  %5 = fir.load %arg0 : !fir.ref<f32>
  %6 = fir.load %arg1 : !fir.ref<f32>
  %7 = fir.call @powf(%5, %6) : (f32, f32) -> f32
  %8 = arith.addf %4, %7 : f32
  %9 = fir.load %arg0 : !fir.ref<f32>
  %10 = fir.load %arg3 : !fir.ref<i32>
  %11 = fir.call @llvm.powi.f32.i32(%9, %10) : (f32, i32) -> f32
  %12 = arith.addf %8, %11 : f32
  fir.store %12 to %0 : !fir.ref<f32>
  %13 = fir.load %0 : !fir.ref<f32>
  return %13 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}, %arg2: !fir.ref<i16> {fir.bindc_name = "s"}, %arg3: !fir.ref<i32> {fir.bindc_name = "i"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg2 : !fir.ref<i16>
  %3 = fir.convert %2 : (i16) -> i32
  %4 = fir.call @llvm.powi.f64.i32(%1, %3) : (f64, i32) -> f64
  %5 = fir.load %arg0 : !fir.ref<f64>
  %6 = fir.load %arg1 : !fir.ref<f64>
  %7 = fir.call @pow(%5, %6) : (f64, f64) -> f64
  %8 = arith.addf %4, %7 : f64
  %9 = fir.load %arg0 : !fir.ref<f64>
  %10 = fir.load %arg3 : !fir.ref<i32>
  %11 = fir.call @llvm.powi.f64.i32(%9, %10) : (f64, i32) -> f64
  %12 = arith.addf %8, %11 : f64
  fir.store %12 to %0 : !fir.ref<f64>
  %13 = fir.load %0 : !fir.ref<f64>
  return %13 : f64
}
func.func private @llvm.powi.f32.i32(f32, i32) -> f32
func.func private @powf(f32, f32) -> f32
func.func private @llvm.powi.f64.i32(f64, i32) -> f64
func.func private @pow(f64, f64) -> f64

//--- exponentiation_integer.fir
// RUN: fir-opt %t/exponentiation_integer.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/exponentiation_integer.fir --check-prefixes="CHECK,CHECK-COMDAT"
// RUN: fir-opt %t/exponentiation_integer.fir --fir-to-llvm-ir="target=x86_64-pc-windows-msvc" | FileCheck %t/exponentiation_integer.fir --check-prefixes="CHECK,CHECK-COMDAT"
// RUN: fir-opt %t/exponentiation_integer.fir --fir-to-llvm-ir="target=aarch64-apple-darwin" | FileCheck %t/exponentiation_integer.fir --check-prefixes="CHECK,CHECK-NOCOMDAT"
// CHECK-COMDAT: llvm.comdat_selector @__mlir_math_ipowi_i32 any
// CHECK-NOCOMDAT-NOT: llvm.comdat_selector @__mlir_math_ipowi_i32 any
// CHECK: @_QPtest_int4
// CHECK: llvm.call @__mlir_math_ipowi_i32({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (i32, i32) -> i32
// CHECK-COMDAT: llvm.func linkonce_odr @__mlir_math_ipowi_i32(%arg0: i32, %arg1: i32) -> i32 comdat(@__llvm_comdat::@__mlir_math_ipowi_i32)
// CHECK-NOCOMDAT: llvm.func linkonce_odr @__mlir_math_ipowi_i32(%arg0: i32, %arg1: i32) -> i32

func.func @_QPtest_int4(%arg0: !fir.ref<i32> {fir.bindc_name = "x"}, %arg1: !fir.ref<i32> {fir.bindc_name = "y"}, %arg2: !fir.ref<i32> {fir.bindc_name = "z"}) {
  %0 = fir.load %arg0 : !fir.ref<i32>
  %1 = fir.load %arg1 : !fir.ref<i32>
  %2 = math.ipowi %0, %1 : i32
  fir.store %2 to %arg2 : !fir.ref<i32>
  return
}

//--- sign_fast.fir
// RUN: fir-opt %t/sign_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sign_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

// CHECK: @_QPtest_real10
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f80, f80) -> f80

// CHECK: @_QPtest_real16
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f128, f128) -> f128

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg1 : !fir.ref<f32>
  %3 = math.copysign %1, %2 : f32
  fir.store %3 to %0 : !fir.ref<f32>
  %4 = fir.load %0 : !fir.ref<f32>
  return %4 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg1 : !fir.ref<f64>
  %3 = math.copysign %1, %2 : f64
  fir.store %3 to %0 : !fir.ref<f64>
  %4 = fir.load %0 : !fir.ref<f64>
  return %4 : f64
}
func.func @_QPtest_real10(%arg0: !fir.ref<f80> {fir.bindc_name = "x"}, %arg1: !fir.ref<f80> {fir.bindc_name = "y"}) -> f80 {
  %0 = fir.alloca f80 {bindc_name = "test_real10", uniq_name = "_QFtest_real10Etest_real10"}
  %1 = fir.load %arg0 : !fir.ref<f80>
  %2 = fir.load %arg1 : !fir.ref<f80>
  %3 = math.copysign %1, %2 : f80
  fir.store %3 to %0 : !fir.ref<f80>
  %4 = fir.load %0 : !fir.ref<f80>
  return %4 : f80
}
func.func @_QPtest_real16(%arg0: !fir.ref<f128> {fir.bindc_name = "x"}, %arg1: !fir.ref<f128> {fir.bindc_name = "y"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "test_real16", uniq_name = "_QFtest_real16Etest_real16"}
  %1 = fir.load %arg0 : !fir.ref<f128>
  %2 = fir.load %arg1 : !fir.ref<f128>
  %3 = math.copysign %1, %2 : f128
  fir.store %3 to %0 : !fir.ref<f128>
  %4 = fir.load %0 : !fir.ref<f128>
  return %4 : f128
}

//--- sign_relaxed.fir
// RUN: fir-opt %t/sign_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sign_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

// CHECK: @_QPtest_real10
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f80, f80) -> f80

// CHECK: @_QPtest_real16
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f128, f128) -> f128

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg1 : !fir.ref<f32>
  %3 = math.copysign %1, %2 : f32
  fir.store %3 to %0 : !fir.ref<f32>
  %4 = fir.load %0 : !fir.ref<f32>
  return %4 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg1 : !fir.ref<f64>
  %3 = math.copysign %1, %2 : f64
  fir.store %3 to %0 : !fir.ref<f64>
  %4 = fir.load %0 : !fir.ref<f64>
  return %4 : f64
}
func.func @_QPtest_real10(%arg0: !fir.ref<f80> {fir.bindc_name = "x"}, %arg1: !fir.ref<f80> {fir.bindc_name = "y"}) -> f80 {
  %0 = fir.alloca f80 {bindc_name = "test_real10", uniq_name = "_QFtest_real10Etest_real10"}
  %1 = fir.load %arg0 : !fir.ref<f80>
  %2 = fir.load %arg1 : !fir.ref<f80>
  %3 = math.copysign %1, %2 : f80
  fir.store %3 to %0 : !fir.ref<f80>
  %4 = fir.load %0 : !fir.ref<f80>
  return %4 : f80
}
func.func @_QPtest_real16(%arg0: !fir.ref<f128> {fir.bindc_name = "x"}, %arg1: !fir.ref<f128> {fir.bindc_name = "y"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "test_real16", uniq_name = "_QFtest_real16Etest_real16"}
  %1 = fir.load %arg0 : !fir.ref<f128>
  %2 = fir.load %arg1 : !fir.ref<f128>
  %3 = math.copysign %1, %2 : f128
  fir.store %3 to %0 : !fir.ref<f128>
  %4 = fir.load %0 : !fir.ref<f128>
  return %4 : f128
}

//--- sign_precise.fir
// RUN: fir-opt %t/sign_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sign_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @copysignf({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f32, f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @copysign({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f64, f64) -> f64

// CHECK: @_QPtest_real10
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @copysignl({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f80, f80) -> f80

// CHECK: @_QPtest_real16
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @llvm.copysign.f128({{%[A-Za-z0-9._]+}}, {{%[A-Za-z0-9._]+}}) : (f128, f128) -> f128

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.load %arg1 : !fir.ref<f32>
  %3 = fir.call @copysignf(%1, %2) : (f32, f32) -> f32
  fir.store %3 to %0 : !fir.ref<f32>
  %4 = fir.load %0 : !fir.ref<f32>
  return %4 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}, %arg1: !fir.ref<f64> {fir.bindc_name = "y"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.load %arg1 : !fir.ref<f64>
  %3 = fir.call @copysign(%1, %2) : (f64, f64) -> f64
  fir.store %3 to %0 : !fir.ref<f64>
  %4 = fir.load %0 : !fir.ref<f64>
  return %4 : f64
}
func.func @_QPtest_real10(%arg0: !fir.ref<f80> {fir.bindc_name = "x"}, %arg1: !fir.ref<f80> {fir.bindc_name = "y"}) -> f80 {
  %0 = fir.alloca f80 {bindc_name = "test_real10", uniq_name = "_QFtest_real10Etest_real10"}
  %1 = fir.load %arg0 : !fir.ref<f80>
  %2 = fir.load %arg1 : !fir.ref<f80>
  %3 = fir.call @copysignl(%1, %2) : (f80, f80) -> f80
  fir.store %3 to %0 : !fir.ref<f80>
  %4 = fir.load %0 : !fir.ref<f80>
  return %4 : f80
}
func.func @_QPtest_real16(%arg0: !fir.ref<f128> {fir.bindc_name = "x"}, %arg1: !fir.ref<f128> {fir.bindc_name = "y"}) -> f128 {
  %0 = fir.alloca f128 {bindc_name = "test_real16", uniq_name = "_QFtest_real16Etest_real16"}
  %1 = fir.load %arg0 : !fir.ref<f128>
  %2 = fir.load %arg1 : !fir.ref<f128>
  %3 = fir.call @llvm.copysign.f128(%1, %2) : (f128, f128) -> f128
  fir.store %3 to %0 : !fir.ref<f128>
  %4 = fir.load %0 : !fir.ref<f128>
  return %4 : f128
}
func.func private @copysignf(f32, f32) -> f32
func.func private @copysign(f64, f64) -> f64
func.func private @copysignl(f80, f80) -> f80
func.func private @llvm.copysign.f128(f128, f128) -> f128

//--- sin_fast.fir
// RUN: fir-opt %t/sin_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sin_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.sin({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.sin({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.sin %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.sin %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- sin_relaxed.fir
// RUN: fir-opt %t/sin_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sin_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.sin({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.intr.sin({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.sin %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.sin %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- sin_precise.fir
// RUN: fir-opt %t/sin_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sin_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sin({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @sinf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @sin(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @sinf(f32) -> f32
func.func private @sin(f64) -> f64

//--- sinh_fast.fir
// RUN: fir-opt %t/sinh_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sinh_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @sinhf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @sinh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @sinhf(f32) -> f32
func.func private @sinh(f64) -> f64

//--- sinh_relaxed.fir
// RUN: fir-opt %t/sinh_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sinh_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @sinhf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @sinh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @sinhf(f32) -> f32
func.func private @sinh(f64) -> f64

//--- sinh_precise.fir
// RUN: fir-opt %t/sinh_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/sinh_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @sinh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @sinhf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @sinh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @sinhf(f32) -> f32
func.func private @sinh(f64) -> f64

//--- tanh_fast.fir
// RUN: fir-opt %t/tanh_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/tanh_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.tanh %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.tanh %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- tanh_relaxed.fir
// RUN: fir-opt %t/tanh_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/tanh_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.tanh %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.tanh %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- tanh_precise.fir
// RUN: fir-opt %t/tanh_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/tanh_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanhf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanh({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @tanhf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @tanh(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @tanhf(f32) -> f32
func.func private @tanh(f64) -> f64

//--- tan_fast.fir
// RUN: fir-opt %t/tan_fast.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/tan_fast.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.tan %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.tan %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- tan_relaxed.fir
// RUN: fir-opt %t/tan_relaxed.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/tan_relaxed.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = math.tan %1 : f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = math.tan %1 : f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}

//--- tan_precise.fir
// RUN: fir-opt %t/tan_precise.fir --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" | FileCheck %t/tan_precise.fir
// CHECK: @_QPtest_real4
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tanf({{%[A-Za-z0-9._]+}}) : (f32) -> f32

// CHECK: @_QPtest_real8
// CHECK: {{%[A-Za-z0-9._]+}} = llvm.call @tan({{%[A-Za-z0-9._]+}}) : (f64) -> f64

func.func @_QPtest_real4(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}) -> f32 {
  %0 = fir.alloca f32 {bindc_name = "test_real4", uniq_name = "_QFtest_real4Etest_real4"}
  %1 = fir.load %arg0 : !fir.ref<f32>
  %2 = fir.call @tanf(%1) : (f32) -> f32
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<f32>
  return %3 : f32
}
func.func @_QPtest_real8(%arg0: !fir.ref<f64> {fir.bindc_name = "x"}) -> f64 {
  %0 = fir.alloca f64 {bindc_name = "test_real8", uniq_name = "_QFtest_real8Etest_real8"}
  %1 = fir.load %arg0 : !fir.ref<f64>
  %2 = fir.call @tan(%1) : (f64) -> f64
  fir.store %2 to %0 : !fir.ref<f64>
  %3 = fir.load %0 : !fir.ref<f64>
  return %3 : f64
}
func.func private @tanf(f32) -> f32
func.func private @tan(f64) -> f64
