// RUN: tco %s | FileCheck %s

// Test peephole optimizations

// CHECK-LABEL: define i8 @test_trunc(
// CHECK-SAME: i256 %[[arg:.*]])
// CHECK-NEXT: = trunc i256 %[[arg]] to i8
// CHECK-NEXT: ret i8
func @test_trunc(%0 : i256) -> i8 {
  %1 = fir.convert %0 : (i256) -> i128
  %2 = fir.convert %1 : (i128) -> i64
  %3 = fir.convert %2 : (i64) -> i32
  %4 = fir.convert %3 : (i32) -> i16
  %5 = fir.convert %4 : (i16) -> i8
  return %5 : i8
}

// CHECK-LABEL: define i256 @test_sext(
// CHECK-SAME: i8 %[[arg:.*]])
// CHECK-NEXT: = sext i8 %[[arg]] to i256
// CHECK-NEXT: ret i256
func @test_sext(%0 : i8) -> i256 {
  %1 = fir.convert %0 : (i8) -> i16
  %2 = fir.convert %1 : (i16) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %4 = fir.convert %3 : (i64) -> i128
  %5 = fir.convert %4 : (i128) -> i256
  return %5 : i256
}

// CHECK-LABEL: define half @test_fptrunc(
// CHECK-SAME: fp128 %[[arg:.*]])
// CHECK-NEXT: %[[res:.*]] = fptrunc fp128 %[[arg]] to half
// CHECK-NEXT: ret half %[[res]]
func @test_fptrunc(%0 : f128) -> f16 {
  %2 = fir.convert %0 : (f128) -> f64
  %3 = fir.convert %2 : (f64) -> f32
  %4 = fir.convert %3 : (f32) -> f16
  return %4 : f16
}

// CHECK-LABEL: define x86_fp80 @test_fpext(
// CHECK-SAME: bfloat %[[arg:.*]])
// CHECK-NEXT: = fpext bfloat %[[arg]] to x86_fp80
// CHECK-NEXT: ret x86_fp80
func @test_fpext(%0 : bf16) -> f80 {
  %2 = fir.convert %0 : (bf16) -> f32
  %3 = fir.convert %2 : (f32) -> f64
  %4 = fir.convert %3 : (f64) -> f80
  return %4 : f80
}

// CHECK-LABEL: define i64 @test_ascending(
// CHECK-SAME: i8 %[[arg:.*]])
// CHECK-NEXT: = sext i8 %[[arg]] to i64
// CHECK-NEXT: ret i64
func @test_ascending(%0 : i8) -> index {
  %1 = fir.convert %0 : (i8) -> i16
  %2 = fir.convert %1 : (i16) -> i32
  %3 = fir.convert %2 : (i32) -> i64
  %5 = fir.convert %3 : (i64) -> index
  return %5 : index
}

// CHECK-LABEL: define i8 @test_descending(
// CHECK-SAME: i64 %[[arg:.*]])
// CHECK-NEXT: = trunc i64 %[[arg]] to i8
// CHECK-NEXT: ret i8
func @test_descending(%0 : index) -> i8 {
  %2 = fir.convert %0 : (index) -> i64
  %3 = fir.convert %2 : (i64) -> i32
  %4 = fir.convert %3 : (i32) -> i16
  %5 = fir.convert %4 : (i16) -> i8
  return %5 : i8
}

// CHECK-LABEL: define float @test_useless(
// CHECK-SAME: float %[[arg:.*]])
// CHECK-NEXT: ret float %[[arg]]
func @test_useless(%0 : f32) -> f32 {
  %1 = fir.convert %0 : (f32) -> f32
  return %1 : f32
}

// CHECK-LABEL: define float @test_useless_sext(
// CHECK-SAME: i32 %[[arg:.*]])
// CHECK-NEXT: %[[res:.*]] = sitofp i32 %[[arg]] to float
// CHECK-NEXT: ret float %[[res]]
func @test_useless_sext(%0 : i32) -> f32 {
  %1 = fir.convert %0 : (i32) -> i64
  %2 = fir.convert %1 : (i64) -> i32
  %3 = fir.convert %2 : (i32) -> f32
  return %3 : f32
}

// CHECK-LABEL: define i16 @test_hump(
// CHECK-SAME: i32 %[[arg:.*]])
// CHECK-NEXT: trunc i32 %[[arg]] to i16
// CHECK-NEXT: ret i16
func @test_hump(%0 : i32) -> i16 {
  %1 = fir.convert %0 : (i32) -> i64
  %2 = fir.convert %1 : (i64) -> i16
  return %2 : i16
}

// CHECK-LABEL: define i16 @test_slump(
// CHECK-SAME: i32 %[[arg:.*]])
// CHECK-NEXT: %[[i:.*]] = trunc i32 %[[arg]] to i8
// CHECK-NEXT: sext i8 %[[i]] to i16
// CHECK-NEXT: ret i16
func @test_slump(%0 : i32) -> i16 {
  %1 = fir.convert %0 : (i32) -> i8
  %2 = fir.convert %1 : (i8) -> i16
  return %2 : i16
}

// CHECK-LABEL: define i64 @test_slump2(
// CHECK-SAME: i64 %[[arg:.*]])
// CHECK-NEXT: %[[i:.*]] = trunc i64 %[[arg]] to i16
// CHECK-NEXT: sext i16 %[[i]] to i64
// CHECK-NEXT: ret i64
func @test_slump2(%0 : index) -> index {
  %1 = fir.convert %0 : (index) -> i16
  %2 = fir.convert %1 : (i16) -> index
  return %2 : index
}
