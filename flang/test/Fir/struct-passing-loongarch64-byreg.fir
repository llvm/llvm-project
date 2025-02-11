/// Test LoongArch64 ABI rewrite of struct passed by value (BIND(C), VALUE derived types).
/// This test test cases where the struct can be passed in registers.
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

// CHECK-LABEL: func.func private @test_func_i16(i64)
func.func private @test_func_i16(%arg0: !ty_i16)
// CHECK-LABEL: func.func @test_call_i16(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<ti16{i:i16}>>) {
func.func @test_call_i16(%arg0: !fir.ref<!ty_i16>) {
  // CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<ti16{i:i16}>>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca i64
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<i64>) -> !fir.ref<!fir.type<ti16{i:i16}>>
  // CHECK: fir.store %[[IN]] to %[[CVT]] : !fir.ref<!fir.type<ti16{i:i16}>>
  // CHECK: %[[LD:.*]] = fir.load %[[ARR]] : !fir.ref<i64>
  %in = fir.load %arg0 : !fir.ref<!ty_i16>
  // CHECK: fir.call @test_func_i16(%[[LD]]) : (i64) -> ()
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  fir.call @test_func_i16(%in) : (!ty_i16) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_func_i32(i64)
func.func private @test_func_i32(%arg0: !ty_i32)

// CHECK-LABEL: func.func private @test_func_i64(i64)
func.func private @test_func_i64(%arg0: !ty_i64)

// CHECK-LABEL: func.func private @test_func_i128(i128)
func.func private @test_func_i128(%arg0: !ty_i128)

// CHECK-LABEL: func.func private @test_func_f16(i64)
func.func private @test_func_f16(%arg0: !ty_f16)

// CHECK-LABEL: func.func private @test_func_f32(f32)
func.func private @test_func_f32(%arg0: !ty_f32)

// CHECK-LABEL: func.func private @test_func_f64(f64)
func.func private @test_func_f64(%arg0: !ty_f64)

// CHECK-LABEL: func.func private @test_func_f128(i128)
func.func private @test_func_f128(%arg0: !ty_f128)

// CHECK-LABEL: func.func private @test_func_bf16(i64)
func.func private @test_func_bf16(%arg0: !ty_bf16)

// CHECK-LABEL: func.func private @test_func_char1(i64)
func.func private @test_func_char1(%arg0: !ty_char1)

// CHECK-LABEL: func.func private @test_func_char2(i64)
func.func private @test_func_char2(%arg0: !ty_char2)

// CHECK-LABEL: func.func private @test_func_log1(i64)
func.func private @test_func_log1(%arg0: !ty_log1)

// CHECK-LABEL: func.func private @test_func_log2(i64)
func.func private @test_func_log2(%arg0: !ty_log2)

// CHECK-LABEL: func.func private @test_func_log4(i64)
func.func private @test_func_log4(%arg0: !ty_log4)

// CHECK-LABEL: func.func private @test_func_log8(i64)
func.func private @test_func_log8(%arg0: !ty_log8)

// CHECK-LABEL: func.func private @test_func_log16(i128)
func.func private @test_func_log16(%arg0: !ty_log16)

// CHECK-LABEL: func.func private @test_func_cmplx_f32(f32, f32)
func.func private @test_func_cmplx_f32(%arg0: !ty_cmplx_f32)

// CHECK-LABEL: func.func private @test_func_cmplx_f64(f64, f64)
func.func private @test_func_cmplx_f64(%arg0: !ty_cmplx_f64)
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
!ty_char1_a8 = !fir.type<t_char_a8{i:!fir.array<8x!fir.char<1>>}>

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "loongarch64-unknown-linux-gnu"} {

// CHECK-LABEL: func.func private @test_func_i32_f32(i32, f32)
func.func private @test_func_i32_f32(%arg0: !ty_i32_f32)
// CHECK-LABEL: func.func @test_call_i32_f32(
// CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>) {
func.func @test_call_i32_f32(%arg0: !fir.ref<!ty_i32_f32>) {
  // CHECK: %[[IN:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>
  // CHECK: %[[STACK:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK: %[[ARR:.*]] = fir.alloca tuple<i32, f32>
  // CHECK: %[[CVT:.*]] = fir.convert %[[ARR]] : (!fir.ref<tuple<i32, f32>>) -> !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>
  // CHECK: fir.store %[[IN]] to %[[CVT]] : !fir.ref<!fir.type<ti32_f32{i:i32,j:f32}>>
  // CHECK: %[[LD:.*]] = fir.load %[[ARR]] : !fir.ref<tuple<i32, f32>>
  // CHECK: %[[VAL_0:.*]] = fir.extract_value %[[LD]], [0 : i32] : (tuple<i32, f32>) -> i32
  // CHECK: %[[VAL_1:.*]] = fir.extract_value %[[LD]], [1 : i32] : (tuple<i32, f32>) -> f32
  %in = fir.load %arg0 : !fir.ref<!ty_i32_f32>
  // CHECK: fir.call @test_func_i32_f32(%[[VAL_0]], %[[VAL_1]]) : (i32, f32) -> ()
  // CHECK: llvm.intr.stackrestore %[[STACK]] : !llvm.ptr
  fir.call @test_func_i32_f32(%in) : (!ty_i32_f32) -> ()
  // CHECK: return
  return
}

// CHECK-LABEL: func.func private @test_func_i32_f64(i32, f64)
func.func private @test_func_i32_f64(%arg0: !ty_i32_f64)

// CHECK-LABEL: func.func private @test_func_i64_f32(i64, f32)
func.func private @test_func_i64_f32(%arg0: !ty_i64_f32)

// CHECK-LABEL: func.func private @test_func_i64_f64(i64, f64)
func.func private @test_func_i64_f64(%arg0: !ty_i64_f64)

// CHECK-LABEL: func.func private @test_func_f64_i64(f64, i64)
func.func private @test_func_f64_i64(%arg0: !ty_f64_i64)

// CHECK-LABEL: func.func private @test_func_f16_f16(i64)
func.func private @test_func_f16_f16(%arg0: !ty_f16_f16)

// CHECK-LABEL: func.func private @test_func_f32_f32(f32, f32)
func.func private @test_func_f32_f32(%arg0: !ty_f32_f32)

// CHECK-LABEL: func.func private @test_func_f64_f64(f64, f64)
func.func private @test_func_f64_f64(%arg0: !ty_f64_f64)

// CHECK-LABEL: func.func private @test_func_f32_i32_i32(!fir.array<2xi64>)
func.func private @test_func_f32_i32_i32(%arg0: !ty_f32_i32_i32)

// CHECK-LABEL: func.func private @test_func_f32_f32_i32(!fir.array<2xi64>)
func.func private @test_func_f32_f32_i32(%arg0: !ty_f32_f32_i32)

// CHECK-LABEL: func.func private @test_func_f32_f32_f32(!fir.array<2xi64>)
func.func private @test_func_f32_f32_f32(%arg0: !ty_f32_f32_f32)

// CHECK-LABEL: func.func private @test_func_i8_a8(i64)
func.func private @test_func_i8_a8(%arg0: !ty_i8_a8)

// CHECK-LABEL: func.func private @test_func_i8_a16(!fir.array<2xi64>)
func.func private @test_func_i8_a16(%arg0: !ty_i8_a16)

// CHECK-LABEL: func.func private @test_func_f32_a2(f32, f32)
func.func private @test_func_f32_a2(%arg0: !ty_f32_a2)

// CHECK-LABEL: func.func private @test_func_f64_a2(f64, f64)
func.func private @test_func_f64_a2(%arg0: !ty_f64_a2)

// CHECK-LABEL: func.func private @test_func_nested_i32_f32(i32, f32)
func.func private @test_func_nested_i32_f32(%arg0: !ty_nested_i32_f32)

// CHECK-LABEL: func.func private @test_func_nested_i8_a8_i32(!fir.array<2xi64>)
func.func private @test_func_nested_i8_a8_i32(%arg0: !ty_nested_i8_a8_i32)


// CHECK: func.func private @not_enough_int_reg_1(i32, i32, i32, i32, i32, i32, i32, i32, i64)
func.func private @not_enough_int_reg_1(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32,
                                        %arg5: i32, %arg6: i32, %arg7: i32, %arg8: !ty_i32_f32)

// CHECK: func.func private @not_enough_int_reg_1b(!fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, i64)
func.func private @not_enough_int_reg_1b(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>, %arg3: !fir.ref<i32>, %arg4: !fir.ref<i32>,
                                         %arg5: !fir.ref<i32>, %arg6: !fir.ref<i32>, %arg7: !fir.ref<i32>, %arg8: !ty_i32_f32)

// CHECK: func.func private @not_enough_int_reg_2(i32, i32, i32, i32, i32, i32, i32, i32, !fir.array<2xi64>)
func.func private @not_enough_int_reg_2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32,
                                        %arg5: i32, %arg6: i32, %arg7: i32, %arg8: !ty_i64_f64)

// CHECK: func.func private @not_enough_fp_reg_1(f32, f32, f32, f32, f32, f32, f32, f32, i64)
func.func private @not_enough_fp_reg_1(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32,
                                        %arg5: f32, %arg6: f32, %arg7: f32, %arg8: !ty_i32_f32)

// CHECK: func.func private @not_enough_fp_reg_1b(!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>, i64)
func.func private @not_enough_fp_reg_1b(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>, %arg2: !fir.ref<f32>, %arg3: !fir.ref<f32>, %arg4: !fir.ref<f32>,
                                        %arg5: !fir.ref<f32>, %arg6: !fir.ref<f32>, %arg7: !fir.ref<f32>, %arg8: !ty_i32_f32)

// CHECK: func.func private @not_enough_fp_reg_2(f32, f32, f32, f32, f32, f32, f32, f32, !fir.array<2xi64>)
func.func private @not_enough_fp_reg_2(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32,
                                        %arg5: f32, %arg6: f32, %arg7: f32, %arg8: !ty_i64_f64)

// CHECK: func.func private @char_not_enough_int_reg(i32, i32, i32, i32, i32, i32, i32, i32, i64)
func.func private @char_not_enough_int_reg(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32,
                                        %arg5: i32, %arg6: i32, %arg7: i32, %arg8: !ty_char1_a8)
}
