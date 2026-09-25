// Test that unsafe-fp-conversion uses plain fptosi/fptoui for float-to-integer
// conversions instead of the saturating intrinsics (llvm.fptosi.sat /
// llvm.fptoui.sat).

// RUN: fir-opt --fir-to-llvm-ir="unsafe-fp-conversion=true" %s | FileCheck %s --check-prefix=UNSAFE
// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s --check-prefix=SAFE

// Test signed conversions from f32.
func.func @float_to_int(%arg0 : f32) -> i32 {
  %0 = fir.convert %arg0 : (f32) -> i32
  return %0 : i32
}

// UNSAFE-LABEL: @float_to_int
// UNSAFE: llvm.fptosi %{{.*}} : f32 to i32
// UNSAFE-NOT: llvm.call_intrinsic "llvm.fptosi.sat"

// SAFE-LABEL: @float_to_int
// SAFE: llvm.call_intrinsic "llvm.fptosi.sat"(%{{.*}}) : (f32) -> i32
// SAFE-NOT: llvm.fptosi %{{.*}} : f32 to i32

// -----

// Test signed conversions from f64.
func.func @double_to_int(%arg0 : f64) -> i32 {
  %0 = fir.convert %arg0 : (f64) -> i32
  return %0 : i32
}

// UNSAFE-LABEL: @double_to_int
// UNSAFE: llvm.fptosi %{{.*}} : f64 to i32
// UNSAFE-NOT: llvm.call_intrinsic "llvm.fptosi.sat"

// SAFE-LABEL: @double_to_int
// SAFE: llvm.call_intrinsic "llvm.fptosi.sat"(%{{.*}}) : (f64) -> i32

// -----

// Test signed conversions from f32 to i64.
func.func @float_to_int8(%arg0 : f32) -> i64 {
  %0 = fir.convert %arg0 : (f32) -> i64
  return %0 : i64
}

// UNSAFE-LABEL: @float_to_int8
// UNSAFE: llvm.fptosi %{{.*}} : f32 to i64
// UNSAFE-NOT: llvm.call_intrinsic "llvm.fptosi.sat"

// SAFE-LABEL: @float_to_int8
// SAFE: llvm.call_intrinsic "llvm.fptosi.sat"(%{{.*}}) : (f32) -> i64

// -----

// Test unsigned conversions from f32.
func.func @float_to_uint(%arg0 : f32) -> ui32 {
  %0 = fir.convert %arg0 : (f32) -> ui32
  return %0 : ui32
}

// UNSAFE-LABEL: @float_to_uint
// UNSAFE: llvm.fptoui %{{.*}} : f32 to i32
// UNSAFE-NOT: llvm.call_intrinsic "llvm.fptoui.sat"

// SAFE-LABEL: @float_to_uint
// SAFE: llvm.call_intrinsic "llvm.fptoui.sat"(%{{.*}}) : (f32) -> i32
