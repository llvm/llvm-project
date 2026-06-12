// Test RISCV64 lp64/lp64d ABI rewrite of struct passed by value (BIND(C), VALUE derived types).

// RUN: fir-opt --target-rewrite="target=riscv64 target-abi=lp64" %s | FileCheck %s --check-prefixes=CHECK-INT
// RUN: fir-opt --target-rewrite="target=riscv64 target-abi=lp64d" %s | FileCheck %s --check-prefixes=CHECK-FLOAT

module attributes {llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "riscv64-unknown-linux-gnu"} {

// ================================================

// CHECK-INT-LABEL: func.func private @single_i64(i64)
// CHECK-INT-LABEL: func.func private @single_i32i32(i64)
// CHECK-INT-LABEL: func.func private @single_i8i8i8(i32)
// CHECK-INT-LABEL: func.func private @single_i8i8i32(i64)
// CHECK-INT-LABEL: func.func private @single_i128(i128)
func.func private @single_i64(!fir.type<single_i64{i:i64}>)
func.func private @single_i32i32(!fir.type<single_i32i32{i:i32,j:i32}>)
func.func private @single_i8i8i8(!fir.type<single_i8i8i8{i:i8,j:i8,k:i8}>)
func.func private @single_i8i8i32(!fir.type<single_i8i8i32{i:i8,j:i8,k:i32}>)
func.func private @single_i128(!fir.type<single_i128{i:i128}>)

// ================================================

// CHECK-INT-LABEL: func.func private @double_i8i64(!fir.array<2xi64>)
// CHECK-INT-LABEL: func.func private @double_complex(!fir.array<2xi64>)
// CHECK-FLOAT-LABEL: func.func private @double_complex(f64, f64)
func.func private @double_i8i64(!fir.type<double_i8i64{i:i8,j:i64}>)
func.func private @double_complex(!fir.type<double_complex{i:complex<f64>}>)

// ================================================

// CHECK-INT-LABEL: func.func private @large_struct(
// CHECK-INT-SAME: !fir.ref<!fir.type<large_struct{i:i64,j:i64,k:i8}>> {{{.*}}, llvm.byval = !fir.type<large_struct{i:i64,j:i64,k:i8}>},
// CHECK-INT-SAME: i64)
// CHECK-INT-LABEL: func.func private @large_bad_alignment(
// CHECK-INT-SAME: !fir.ref<!fir.type<large_bad_alignment{i:i8,j:i64,k:i8}>> {{{.*}}, llvm.byval = !fir.type<large_bad_alignment{i:i8,j:i64,k:i8}>})
func.func private @large_struct(!fir.type<large_struct{i:i64,j:i64,k:i8}>, i64)
func.func private @large_bad_alignment(!fir.type<large_bad_alignment{i:i8,j:i64,k:i8}>)

// ================================================

// CHECK-INT-LABEL:   func.func private @single_float(i32)
// CHECK-INT-LABEL:   func.func private @float_int(i64)
// CHECK-INT-LABEL:   func.func private @float_char(i64)
// CHECK-INT-LABEL:   func.func private @logical_double(!fir.array<2xi64>)
// CHECK-FLOAT-LABEL: func.func private @single_float(f32)
// CHECK-FLOAT-LABEL: func.func private @float_int(f32, i32)
// CHECK-FLOAT-LABEL: func.func private @float_char(f32, i8)
// CHECK-FLOAT-LABEL: func.func private @logical_double(i8, f64)
func.func private @single_float(!fir.type<single_float{f:f32}>)
func.func private @float_int(!fir.type<float_int{f:f32,i:i32}>)
func.func private @float_char(!fir.type<float_char{f:f32,c:!fir.char<1>}>)
func.func private @logical_double(!fir.type<logical_double{l:!fir.logical<1>,f:f64}>)

// ================================================

// CHECK-INT-LABEL:   func.func private @split_struct(i64, i64, i64, i64, i64, i64, i64, !fir.array<2xi64>, i64)
// CHECK-FLOAT-LABEL: func.func private @split_struct(i64, i64, i64, i64, i64, i64, i64, !fir.array<2xi64>, i64)
func.func private @split_struct(i64, i64, i64, i64, i64, i64, i64, !fir.type<split_struct{i:i8,j:i64}>, i64)

// CHECK-INT-LABEL:   func.func private @split_struct_float(f64, f64, f64, f64, f64, f64, f64, !fir.array<2xi64>)
// CHECK-FLOAT-LABEL: func.func private @split_struct_float(f64, f64, f64, f64, f64, f64, f64, f32, i64)
func.func private @split_struct_float(f64, f64, f64, f64, f64, f64, f64, !fir.type<split_struct_float{i:f32,j:i64}>)

// CHECK-INT-LABEL:   func.func private @split_struct_float2(f64, f64, f64, f64, f64, f64, f64, !fir.array<2xi64>)
// CHECK-FLOAT-LABEL: func.func private @split_struct_float2(f64, f64, f64, f64, f64, f64, f64, !fir.array<2xi64>)
func.func private @split_struct_float2(f64, f64, f64, f64, f64, f64, f64, !fir.type<split_struct_float2{i:f64,j:f32}>)

// CHECK-INT-LABEL:   func.func private @split_struct_float3(f64, f64, f64, f64, f64, f64, f64, !fir.array<2xi64>, !fir.array<2xi64>)
// CHECK-FLOAT-LABEL: func.func private @split_struct_float3(f64, f64, f64, f64, f64, f64, f64, !fir.array<2xi64>, f64, i32)
func.func private @split_struct_float3(f64, f64, f64, f64, f64, f64, f64, !fir.type<split_struct_float3{i:f64,j:f32}>, !fir.type<split_struct_mixed{i:f64,j:i32}>)

// CHECK-INT-LABEL:   func.func private @split_struct_float4(f64, f64, f64, f64, f64, f64, f64, !fir.array<2xi64>, !fir.array<2xi64>)
// CHECK-FLOAT-LABEL: func.func private @split_struct_float4(f64, f64, f64, f64, f64, f64, f64, i64, f32, !fir.array<2xi64>)
func.func private @split_struct_float4(f64, f64, f64, f64, f64, f64, f64, !fir.type<split_struct_float4{i:i64,j:f32}>, !fir.type<split_struct_mixed{i:f64,j:i32}>)

// CHECK-INT-LABEL:   func.func private @no_remaining_gpr(i64, i64, i64, i64, i64, i64, i64, i64, i64)
// CHECK-FLOAT-LABEL: func.func private @no_remaining_gpr(i64, i64, i64, i64, i64, i64, i64, i64, i64)
func.func private @no_remaining_gpr(i64, i64, i64, i64, i64, i64, i64, i64, !fir.type<no_remaining_gpr{i:f32,j:i32}>)

// CHECK-INT-LABEL:   func.func private @no_remaining_gpr_int128(i64, i64, i64, i64, i64, i64, i128, i64)
// CHECK-FLOAT-LABEL: func.func private @no_remaining_gpr_int128(i64, i64, i64, i64, i64, i64, i128, i64)
func.func private @no_remaining_gpr_int128(i64, i64, i64, i64, i64, i64, i128, !fir.type<no_remaining_gpr{i:f32,j:i32}>)

// CHECK-INT-LABEL:   func.func private @single_remaining_gpr(i64, i64, i64, i64, i64, i64, i64, i64)
// CHECK-FLOAT-LABEL: func.func private @single_remaining_gpr(i64, i64, i64, i64, i64, i64, i64, f32, i32)
func.func private @single_remaining_gpr(i64, i64, i64, i64, i64, i64, i64, !fir.type<single_remaining_gpr{i:f32,j:i32}>)

// CHECK-INT-LABEL:   func.func private @prev_struct_passed(!fir.array<2xi64>, i32)
// CHECK-FLOAT-LABEL: func.func private @prev_struct_passed(!fir.array<2xi64>, i32)
func.func private @prev_struct_passed(!fir.type<prev{i:i64,j:i64}>, !fir.type<current{i:i32}>)

}
