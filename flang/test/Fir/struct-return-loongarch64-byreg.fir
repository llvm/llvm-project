/// Test LoongArch64 ABI rewrite of struct returned by value (BIND(C), VALUE derived types).
/// This test test cases where the struct can be returned in registers.
/// Test cases can be roughly divided into two categories:
///  - struct with a single intrinsic component;
///  - sturct with more than one field;
/// Since the argument marshalling logic is largely the same within each category,
/// only the first example in each category checks the entire invocation process,
/// while the other examples only check the signatures.

// REQUIRES: loongarch-registered-target
// RUN: fir-opt --split-input-file --target-rewrite="target=loongarch64-unknown-linux-gnu" %s | FileCheck %s


/// *********************** Struct with a single intrinsic component *********************** ///

!ty_i16   = !fir.type<ti16{i:i16}>
!ty_i32   = !fir.type<ti32{i:i32}>
!ty_i64   = !fir.type<ti64{i:i64}>
!ty_i128  = !fir.type<ti128{i:i128}>
!ty_f16   = !fir.type<tf16{i:f16}>
!ty_f32   = !fir.type<tf32{i:f32}>
!ty_f64   = !fir.type<tf64{i:f64}>
!ty_f128  = !fir.type<tf128{i:f128}>
!ty_bf16  = !fir.type<tbf16{i:bf16}>
!ty_char1 = !fir.type<tchar1{i:!fir.char<1>}>
!ty_char2 = !fir.type<tchar2{i:!fir.char<1,2>}>
!ty_log1  = !fir.type<tlog1{i:!fir.logical<1>}>
!ty_log2  = !fir.type<tlog2{i:!fir.logical<2>}>
!ty_log4  = !fir.type<tlog4{i:!fir.logical<4>}>
!ty_log8  = !fir.type<tlog8{i:!fir.logical<8>}>
!ty_log16 = !fir.type<tlog16{i:!fir.logical<16>}>
!ty_cmplx_f32  = !fir.type<tcmplx_f32{i:complex<f32>}>
!ty_cmplx_f64  = !fir.type<tcmplx_f64{i:complex<f64>}>

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "loongarch64-unknown-linux-gnu"} {

// CHECK-LABEL: func.func private @test_func_i16() -> i64
func.func private @test_func_i16() -> !ty_i16
// CHECK-LABEL: func.func @test_call_i16(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<ti16{i:i16}>>) {
func.func @test_call_i16(%arg0: !fir.ref<!ty_i16>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_func_i16() : () -> i64
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca i64
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<i64>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<i64>) -> !fir.ref<!fir.type<ti16{i:i16}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<ti16{i:i16}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_func_i16() : () -> !ty_i16
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<ti16{i:i16}>>
  fir.store %out to %arg0 : !fir.ref<!ty_i16>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_func_i32() -> i64
func.func private @test_func_i32() -> !ty_i32

// CHECK-LABEL: func.func private @test_func_i64() -> i64
func.func private @test_func_i64() -> !ty_i64

// CHECK-LABEL: func.func private @test_func_i128() -> i128
func.func private @test_func_i128() -> !ty_i128

// CHECK-LABEL: func.func private @test_func_f16() -> i64
func.func private @test_func_f16() -> !ty_f16

// CHECK-LABEL: func.func private @test_func_f32() -> f32
func.func private @test_func_f32() -> !ty_f32

// CHECK-LABEL: func.func private @test_func_f64() -> f64
func.func private @test_func_f64() -> !ty_f64

// CHECK-LABEL: func.func private @test_func_f128() -> i128
func.func private @test_func_f128() -> !ty_f128

// CHECK-LABEL: func.func private @test_func_bf16() -> i64
func.func private @test_func_bf16() -> !ty_bf16

// CHECK-LABEL: func.func private @test_func_char1() -> i64
func.func private @test_func_char1() -> !ty_char1

// CHECK-LABEL: func.func private @test_func_char2() -> i64
func.func private @test_func_char2() -> !ty_char2

// CHECK-LABEL: func.func private @test_func_log1() -> i64
func.func private @test_func_log1() -> !ty_log1

// CHECK-LABEL: func.func private @test_func_log2() -> i64
func.func private @test_func_log2() -> !ty_log2

// CHECK-LABEL: func.func private @test_func_log4() -> i64
func.func private @test_func_log4() -> !ty_log4

// CHECK-LABEL: func.func private @test_func_log8() -> i64
func.func private @test_func_log8() -> !ty_log8

// CHECK-LABEL: func.func private @test_func_log16() -> i128
func.func private @test_func_log16() -> !ty_log16

// CHECK-LABEL: func.func private @test_func_cmplx_f32() -> tuple<f32, f32>
func.func private @test_func_cmplx_f32() -> !ty_cmplx_f32

// CHECK-LABEL: func.func private @test_func_cmplx_f64() -> tuple<f64, f64>
func.func private @test_func_cmplx_f64() -> !ty_cmplx_f64
}


/// *************************** Struct with more than one field **************************** ///

// -----

!ty_i32_f32 = !fir.type<ti32_f32{i:i32,j:f32}>
!ty_i32_f64 = !fir.type<ti32_f64{i:i32,j:f64}>
!ty_i64_f32 = !fir.type<ti64_f32{i:i64,j:f32}>
!ty_i64_f64 = !fir.type<ti64_f64{i:i64,j:f64}>
!ty_f64_i64 = !fir.type<tf64_i64{i:f64,j:i64}>
!ty_f16_f16 = !fir.type<tf16_f16{i:f16,j:f16}>
!ty_f32_f32 = !fir.type<tf32_f32{i:f32,j:f32}>
!ty_f64_f64 = !fir.type<tf64_f64{i:f64,j:f64}>
!ty_f32_i32_i32 = !fir.type<tf32_i32_i32{i:f32,j:i32,k:i32}>
!ty_f32_f32_i32 = !fir.type<tf32_f32_i32{i:f32,j:f32,k:i32}>
!ty_f32_f32_f32 = !fir.type<tf32_f32_f32{i:f32,j:f32,k:f32}>

!ty_i8_a8  = !fir.type<ti8_a8{i:!fir.array<8xi8>}>
!ty_i8_a16 = !fir.type<ti8_a16{i:!fir.array<16xi8>}>
!ty_f32_a2 = !fir.type<tf32_a2{i:!fir.array<2xf32>}>
!ty_f64_a2 = !fir.type<tf64_a2{i:!fir.array<2xf64>}>
!ty_nested_i32_f32 = !fir.type<t11{i:!ty_i32_f32}>
!ty_nested_i8_a8_i32 = !fir.type<t12{i:!ty_i8_a8, j:i32}>

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "loongarch64-unknown-linux-gnu"} {

// CHECK-LABEL: func.func private @test_func_i32_f32() -> tuple<i32, f32>
func.func private @test_func_i32_f32() -> !ty_i32_f32
// CHECK-LABEL: func.func @test_call_i32_f32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>) {
func.func @test_call_i32_f32(%arg0: !fir.ref<!ty_i32_f32>) {
  // CHECK: %[[OUT:.*]] = fir.call @test_func_i32_f32() : () -> tuple<i32, f32>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca tuple<i32, f32>
  // CHECK: fir.store %[[OUT]] to %[[ARR]] : !fir.ref<tuple<i32, f32>>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<tuple<i32, f32>>) -> !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>
  // CHECK: %[[LD:.*]] = fir.load %[[CVT]] : !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  %out = fir.call @test_func_i32_f32() : () -> !ty_i32_f32
  // CHECK: fir.store %[[LD]] to %[[ARG0]] : !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>
  fir.store %out to %arg0 : !fir.ref<!ty_i32_f32>
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_func_i32_f64() -> tuple<i32, f64>
func.func private @test_func_i32_f64() -> !ty_i32_f64

// CHECK-LABEL: func.func private @test_func_i64_f32() -> tuple<i64, f32>
func.func private @test_func_i64_f32() -> !ty_i64_f32

// CHECK-LABEL: func.func private @test_func_i64_f64() -> tuple<i64, f64>
func.func private @test_func_i64_f64() -> !ty_i64_f64

// CHECK-LABEL: func.func private @test_func_f64_i64() -> tuple<f64, i64>
func.func private @test_func_f64_i64() -> !ty_f64_i64

// CHECK-LABEL: func.func private @test_func_f16_f16() -> i64
func.func private @test_func_f16_f16() -> !ty_f16_f16

// CHECK-LABEL: func.func private @test_func_f32_f32() -> tuple<f32, f32>
func.func private @test_func_f32_f32() -> !ty_f32_f32

// CHECK-LABEL: func.func private @test_func_f64_f64() -> tuple<f64, f64>
func.func private @test_func_f64_f64() -> !ty_f64_f64

// CHECK-LABEL: func.func private @test_func_f32_i32_i32() -> !fir.array<2xi64>
func.func private @test_func_f32_i32_i32() -> !ty_f32_i32_i32

// CHECK-LABEL: func.func private @test_func_f32_f32_i32() -> !fir.array<2xi64>
func.func private @test_func_f32_f32_i32() -> !ty_f32_f32_i32

// CHECK-LABEL: func.func private @test_func_f32_f32_f32() -> !fir.array<2xi64>
func.func private @test_func_f32_f32_f32() -> !ty_f32_f32_f32

// CHECK-LABEL: func.func private @test_func_i8_a8() -> i64
func.func private @test_func_i8_a8() -> !ty_i8_a8

// CHECK-LABEL: func.func private @test_func_i8_a16() -> !fir.array<2xi64>
func.func private @test_func_i8_a16() -> !ty_i8_a16

// CHECK-LABEL: func.func private @test_func_f32_a2() -> tuple<f32, f32>
func.func private @test_func_f32_a2() -> !ty_f32_a2

// CHECK-LABEL: func.func private @test_func_f64_a2() -> tuple<f64, f64>
func.func private @test_func_f64_a2() -> !ty_f64_a2

// CHECK-LABEL: func.func private @test_func_nested_i32_f32() -> tuple<i32, f32>
func.func private @test_func_nested_i32_f32() -> !ty_nested_i32_f32

// CHECK-LABEL: func.func private @test_func_nested_i8_a8_i32() -> !fir.array<2xi64>
func.func private @test_func_nested_i8_a8_i32() -> !ty_nested_i8_a8_i32
}
