// Test RISCV64 lp64/lp64d ABI rewrite of struct returned by value (BIND(C), VALUE derived types).

// RUN: fir-opt --target-rewrite="target=riscv64 target-abi=lp64" %s | FileCheck --check-prefixes=CHECK-INT %s
// RUN: fir-opt --target-rewrite="target=riscv64 target-abi=lp64d" %s | FileCheck --check-prefixes=CHECK-FLOAT %s

!single_struct = !fir.type<t1{i:i32,j:i16}>
!double_struct = !fir.type<t2{i:i8,j:i32,k:i16}>
!too_big = !fir.type<t3{i:i64,j:i64,k:i8}>
!complex_double = !fir.type<t4{i:complex<f64>}>
!mixed_float = !fir.type<t5{i:f32,j:i32}>
!large_scalar = !fir.type<t6{i:i128}>

module attributes {llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "riscv64-unknown-linux-gnu"} {

func.func private @single_return() -> !single_struct
func.func @test_call_single_return(%arg0 : !fir.ref<!single_struct>) {
  %out = fir.call @single_return() : () -> !single_struct
  fir.store %out to %arg0 : !fir.ref<!single_struct>
  return
}

// CHECK-INT-LABEL:   func.func private @single_return() -> i64
// CHECK-INT-LABEL:   func.func @test_call_single_return(
// CHECK-INT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t1{i:i32,j:i16}>>) {
// CHECK-INT:          %[[CALL_0:.*]] = fir.call @single_return() : () -> i64
// CHECK-INT:          %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT:          %[[ALLOCA_0:.*]] = fir.alloca i64
// CHECK-INT:          fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<i64>
// CHECK-INT:          %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<i64>) -> !fir.ref<!fir.type<t1{i:i32,j:i16}>>
// CHECK-INT:          %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t1{i:i32,j:i16}>>
// CHECK-INT:          llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT:          fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t1{i:i32,j:i16}>>
// CHECK-INT:          return
// CHECK-INT:        }

// ================================================

func.func private @double_return() -> !double_struct
func.func @test_call_double_return(%arg0 : !fir.ref<!double_struct>) {
  %out = fir.call @double_return() : () -> !double_struct
  fir.store %out to %arg0 : !fir.ref<!double_struct>
  return
}

// CHECK-INT-LABEL:   func.func private @double_return() -> !fir.array<2xi64>
// CHECK-INT-LABEL:   func.func @test_call_double_return(
// CHECK-INT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t2{i:i8,j:i32,k:i16}>>) {
// CHECK-INT:          %[[CALL_0:.*]] = fir.call @double_return() : () -> !fir.array<2xi64>
// CHECK-INT:          %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT:          %[[ALLOCA_0:.*]] = fir.alloca !fir.array<2xi64>
// CHECK-INT:          fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.array<2xi64>>
// CHECK-INT:          %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.array<2xi64>>) -> !fir.ref<!fir.type<t2{i:i8,j:i32,k:i16}>>
// CHECK-INT:          %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t2{i:i8,j:i32,k:i16}>>
// CHECK-INT:          llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT:          fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t2{i:i8,j:i32,k:i16}>>
// CHECK-INT:          return

// ================================================

func.func private @too_big_return() -> !too_big
func.func @test_call_too_big_return(%arg0 : !fir.ref<!too_big>) {
  %out = fir.call @too_big_return() : () -> !too_big
  fir.store %out to %arg0 : !fir.ref<!too_big>
  return
}

// CHECK-INT-LABEL:   func.func private @too_big_return(!fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t3{i:i64,j:i64,k:i8}>})
// CHECK-INT-LABEL:   func.func @test_call_too_big_return(
// CHECK-INT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>) {
// CHECK-INT           %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT           %[[ALLOCA_0:.*]] = fir.alloca !fir.type<t3{i:i64,j:i64,k:i8}>
// CHECK-INT           fir.call @too_big_return(%[[ALLOCA_0]]) : (!fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t3{i:i64,j:i64,k:i8}>}) -> ()
// CHECK-INT           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>) -> !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>
// CHECK-INT           %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>
// CHECK-INT           llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT           fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>
// CHECK-INT           return


// ================================================

func.func private @large_scalar_return() -> !large_scalar
func.func @test_call_large_scalar_return(%arg0 : !fir.ref<!large_scalar>) {
  %out = fir.call @large_scalar_return() : () -> !large_scalar
  fir.store %out to %arg0 : !fir.ref<!large_scalar>
  return
}

// CHECK-INT-LABEL:   func.func private @large_scalar_return() -> i128
// CHECK-INT-LABEL:   func.func @test_call_large_scalar_return(
// CHECK-INT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t6{i:i128}>>) {
// CHECK-INT:           %[[CALL_0:.*]] = fir.call @large_scalar_return() : () -> i128
// CHECK-INT:           %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT:           %[[ALLOCA_0:.*]] = fir.alloca i128
// CHECK-INT:           fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<i128>
// CHECK-INT:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<i128>) -> !fir.ref<!fir.type<t6{i:i128}>>
// CHECK-INT:           %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t6{i:i128}>>
// CHECK-INT:           llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT:           fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t6{i:i128}>>
// CHECK-INT:           return
// CHECK-INT:         }

// ================================================

func.func private @too_big_more_args(i64) -> !too_big
func.func @test_call_too_big_more_args(%arg0 : i64, %arg1 : !fir.ref<!too_big>) {
  %out = fir.call @too_big_more_args(%arg0) : (i64) -> !too_big
  fir.store %out to %arg1 : !fir.ref<!too_big>
  return
}

// CHECK-INT-LABEL:   func.func private @too_big_more_args(!fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t3{i:i64,j:i64,k:i8}>}, i64)
// CHECK-INT-LABEL:   func.func @test_call_too_big_more_args(
// CHECK-INT-SAME:      %[[ARG0:.*]]: i64,
// CHECK-INT-SAME:      %[[ARG1:.*]]: !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>) {
// CHECK-INT           %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT           %[[ALLOCA_0:.*]] = fir.alloca !fir.type<t3{i:i64,j:i64,k:i8}>
// CHECK-INT           fir.call @too_big_more_args(%[[ALLOCA_0]], %[[ARG0]]) : (!fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t3{i:i64,j:i64,k:i8}>}, i64) -> ()
// CHECK-INT           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>) -> !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>
// CHECK-INT           %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>
// CHECK-INT           llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT           fir.store %[[LOAD_0]] to %[[ARG1]] : !fir.ref<!fir.type<t3{i:i64,j:i64,k:i8}>>
// CHECK-INT           return


// ================================================

func.func private @complex_type() -> !complex_double
func.func @test_complex_type(%arg0 : !fir.ref<!complex_double>) {
  %out = fir.call @complex_type() : () -> !complex_double
  fir.store %out to %arg0 : !fir.ref<!complex_double>
  return
}


// CHECK-FLOAT-LABEL:   func.func @test_complex_type(
// CHECK-FLOAT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t4{i:complex<f64>}>>) {
// CHECK-FLOAT:           %[[CALL_0:.*]] = fir.call @complex_type() : () -> tuple<f64, f64>
// CHECK-FLOAT:           %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-FLOAT:           %[[ALLOCA_0:.*]] = fir.alloca tuple<f64, f64>
// CHECK-FLOAT:           fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<tuple<f64, f64>>
// CHECK-FLOAT:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<tuple<f64, f64>>) -> !fir.ref<!fir.type<t4{i:complex<f64>}>>
// CHECK-FLOAT:           %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t4{i:complex<f64>}>>
// CHECK-FLOAT:           llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-FLOAT:           fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t4{i:complex<f64>}>>
// CHECK-FLOAT:           return

// CHECK-INT-LABEL:   func.func @test_complex_type(
// CHECK-INT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t4{i:complex<f64>}>>) {
// CHECK-INT:           %[[CALL_0:.*]] = fir.call @complex_type() : () -> !fir.array<2xi64>
// CHECK-INT:           %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<2xi64>
// CHECK-INT:           fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.array<2xi64>>
// CHECK-INT:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.array<2xi64>>) -> !fir.ref<!fir.type<t4{i:complex<f64>}>>
// CHECK-INT:           %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t4{i:complex<f64>}>>
// CHECK-INT:           llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT:           fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t4{i:complex<f64>}>>
// CHECK-INT:           return


// ================================================

func.func private @mixed_float_type() -> !mixed_float
func.func @test_mixed_float_type(%arg0 : !fir.ref<!mixed_float>) {
  %out = fir.call @mixed_float_type() : () -> !mixed_float
  fir.store %out to %arg0 : !fir.ref<!mixed_float>
  return
}

// CHECK-FLOAT-LABEL:   func.func @test_mixed_float_type(
// CHECK-FLOAT-SAME:     %[[ARG0:.*]]: !fir.ref<!fir.type<t5{i:f32,j:i32}>>) {
// CHECK-FLOAT:          %[[CALL_0:.*]] = fir.call @mixed_float_type() : () -> tuple<f32, i32>
// CHECK-FLOAT:          %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-FLOAT:          %[[ALLOCA_0:.*]] = fir.alloca tuple<f32, i32>
// CHECK-FLOAT:          fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<tuple<f32, i32>>
// CHECK-FLOAT:          %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<tuple<f32, i32>>) -> !fir.ref<!fir.type<t5{i:f32,j:i32}>>
// CHECK-FLOAT:          %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t5{i:f32,j:i32}>>
// CHECK-FLOAT:          llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-FLOAT:          fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t5{i:f32,j:i32}>>
// CHECK-FLOAT:          return

// CHECK-INT-LABEL:   func.func @test_mixed_float_type(
// CHECK-INT-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<t5{i:f32,j:i32}>>) {
// CHECK-INT:           %[[CALL_0:.*]] = fir.call @mixed_float_type() : () -> i64
// CHECK-INT:           %[[INTR_0:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK-INT:           %[[ALLOCA_0:.*]] = fir.alloca i64
// CHECK-INT:           fir.store %[[CALL_0]] to %[[ALLOCA_0]] : !fir.ref<i64>
// CHECK-INT:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<i64>) -> !fir.ref<!fir.type<t5{i:f32,j:i32}>>
// CHECK-INT:           %[[LOAD_0:.*]] = fir.load %[[CONVERT_0]] : !fir.ref<!fir.type<t5{i:f32,j:i32}>>
// CHECK-INT:           llvm.intr.stackrestore %[[INTR_0]] : !llvm.ptr
// CHECK-INT:           fir.store %[[LOAD_0]] to %[[ARG0]] : !fir.ref<!fir.type<t5{i:f32,j:i32}>>
// CHECK-INT:           return

}
