// Test AArch64 ABI rewrite of struct passed by value (BIND(C), VALUE derived types).
// RUN: fir-opt --target-rewrite="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

// CHECK-LABEL: func.func private @small_i32(!fir.array<2xi64>)
func.func private @small_i32(!fir.type<small_i32{i:i32,j:i32,k:i32}>)
// CHECK-LABEL: func.func private @small_i64(!fir.array<2xi64>)
func.func private @small_i64(!fir.type<small_i64{i:i64,j:i64}>)
// CHECK-LABEL: func.func private @small_mixed(!fir.array<2xi64>)
func.func private @small_mixed(!fir.type<small_mixed{i:i64,j:f32,k:i32}>)
// CHECK-LABEL: func.func private @small_non_hfa(!fir.array<2xi64>)
func.func private @small_non_hfa(!fir.type<small_non_hfa{i:f64,j:f32,k:f16}>)

// CHECK-LABEL: func.func private @hfa_f16(!fir.type<hfa_f16{i:f16,j:f16}>)
func.func private @hfa_f16(!fir.type<hfa_f16{i:f16,j:f16}>)
// CHECK-LABEL: func.func private @hfa_bf16(!fir.type<hfa_bf16{i:bf16,j:bf16,k:bf16,l:bf16}>)
func.func private @hfa_bf16(!fir.type<hfa_bf16{i:bf16,j:bf16,k:bf16,l:bf16}>)
// CHECK-LABEL: func.func private @hfa_f32(!fir.type<hfa_f32{i:f32,j:f32}>)
func.func private @hfa_f32(!fir.type<hfa_f32{i:f32,j:f32}>)
// CHECK-LABEL: func.func private @hfa_f64(!fir.type<hfa_f64{i:f64,j:f64,k:f64}>)
func.func private @hfa_f64(!fir.type<hfa_f64{i:f64,j:f64,k:f64}>)
// CHECK-LABEL: func.func private @hfa_f128(!fir.type<hfa_f128{i:f128,j:f128,k:f128,l:f128}>)
func.func private @hfa_f128(!fir.type<hfa_f128{i:f128,j:f128,k:f128,l:f128}>)

// CHECK-LABEL: func.func private @multi_small_integer(!fir.array<2xi64>, !fir.array<2xi64>)
func.func private @multi_small_integer(!fir.type<small_i32{i:i32,j:i32,k:i32}>, !fir.type<small_i64{i:i64,j:i64}>)
// CHECK-LABEL: func.func private @multi_hfas(!fir.type<hfa_f16{i:f16,j:f16}>, !fir.type<hfa_f128{i:f128,j:f128,k:f128,l:f128}>)
func.func private @multi_hfas(!fir.type<hfa_f16{i:f16,j:f16}>, !fir.type<hfa_f128{i:f128,j:f128,k:f128,l:f128}>)
// CHECK-LABEL: func.func private @multi_mixed(!fir.type<hfa_f64{i:f64,j:f64,k:f64}>, !fir.array<2xi64>, !fir.type<hfa_f32{i:f32,j:f32}>, !fir.array<2xi64>)
func.func private @multi_mixed(!fir.type<hfa_f64{i:f64,j:f64,k:f64}>,!fir.type<small_non_hfa{i:f64,j:f32,k:f16}>,!fir.type<hfa_f32{i:f32,j:f32}>,!fir.type<small_i64{i:i64,j:i64}>)

// CHECK-LABEL: func.func private @int_max(!fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>)
func.func private @int_max(!fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>)
// CHECK-LABEL: func.func private @hfa_max(!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>, !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>)
func.func private @hfa_max(!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>, !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>)
// CHECK-LABEL: func.func private @max(!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
// CHECK-SAME: !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>)
func.func private @max(!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
                       !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>)


// CHECK-LABEL: func.func private @too_many_int(!fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.array<2xi64>,
// CHECK-SAME: !fir.ref<!fir.type<int_max{i:i64,j:i64}>> {{{.*}}, llvm.byval = !fir.type<int_max{i:i64,j:i64}>})
func.func private @too_many_int(!fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>,
                       !fir.type<int_max{i:i64,j:i64}>)
// CHECK-LABEL: func.func private @too_many_hfa(!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
// CHECK-SAME: !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
// CHECK-SAME: !fir.ref<!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>> {{{.*}}, llvm.byval = !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>})
func.func private @too_many_hfa(!fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
                           !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>,
                           !fir.type<hfa_max{i:f128,j:f128,k:f128,l:f128}>)

// CHECK-LABEL: func.func private @too_big(!fir.ref<!fir.type<too_big{i:!fir.array<5xi32>}>> {{{.*}}, llvm.byval = !fir.type<too_big{i:!fir.array<5xi32>}>})
func.func private @too_big(!fir.type<too_big{i:!fir.array<5xi32>}>)
