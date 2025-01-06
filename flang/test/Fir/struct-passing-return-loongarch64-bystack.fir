/// Test LoongArch64 ABI rewrite of struct passed and returned by value (BIND(C), VALUE derived types).
/// This test test cases where the struct must be passed or returned on the stack.

// REQUIRES: loongarch-registered-target
// RUN: tco --target=loongarch64-unknown-linux-gnu %s | FileCheck %s

!ty_int_toobig = !fir.type<int_toobig{i:!fir.array<5xi32>}>
!ty_int_toobig_align16 = !fir.type<int_toobig_align16{i:i128,j:i8}>
!ty_fp_toobig = !fir.type<fp_toobig{i:!fir.array<5xf64>}>
!ty_fp_toobig_align16 = !fir.type<fp_toobig_align16{i:f128,j:f32}>

!ty_i32_f32 = !fir.type<i32_f32{i:i32,j:f32}>
!ty_nested_toobig = !fir.type<nested_toobig{i:!fir.array<3x!ty_i32_f32>}>
!ty_badly_aligned = !fir.type<badly_aligned{i:f32,j:f64,k:f32}>
!ty_logical_toobig = !fir.type<logical_toobig{i:!fir.array<17x!fir.logical<1>>}>
!ty_cmplx_toobig = !fir.type<cmplx_toobig{i:!fir.array<4xcomplex<f32>>}>
!ty_char_toobig = !fir.type<char_toobig{i:!fir.array<17x!fir.char<1>>}>
!ty_cmplx_f128 = !fir.type<cmplx_f128{i:complex<f128>}>

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "loongarch64-unknown-linux-gnu"} {

// CHECK: declare void @takes_int_toobig(ptr byval(%int_toobig) align 4)
func.func private @takes_int_toobig(%arg0: !ty_int_toobig) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_int_toobig(ptr sret(%int_toobig) align 4)
func.func private @return_int_toobig() -> !ty_int_toobig attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_int_toobig_align16(ptr byval(%int_toobig_align16) align 16)
func.func private @takes_int_toobig_align16(%arg0: !ty_int_toobig_align16) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_int_toobig_align16(ptr sret(%int_toobig_align16) align 16)
func.func private @return_int_toobig_align16() -> !ty_int_toobig_align16 attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_fp_toobig(ptr byval(%fp_toobig) align 8)
func.func private @takes_fp_toobig(%arg0: !ty_fp_toobig) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_fp_toobig(ptr sret(%fp_toobig) align 8)
func.func private @return_fp_toobig() -> !ty_fp_toobig attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_fp_toobig_align16(ptr byval(%fp_toobig_align16) align 16)
func.func private @takes_fp_toobig_align16(%arg0: !ty_fp_toobig_align16) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_fp_toobig_align16(ptr sret(%fp_toobig_align16) align 16)
func.func private @return_fp_toobig_align16() -> !ty_fp_toobig_align16 attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_nested_toobig(ptr byval(%nested_toobig) align 4)
func.func private @takes_nested_toobig(%arg0: !ty_nested_toobig) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_nested_toobig(ptr sret(%nested_toobig) align 4)
func.func private @return_nested_toobig() -> !ty_nested_toobig attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_badly_aligned(ptr byval(%badly_aligned) align 8)
func.func private @takes_badly_aligned(%arg0: !ty_badly_aligned) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_badly_aligned(ptr sret(%badly_aligned) align 8)
func.func private @return_badly_aligned() -> !ty_badly_aligned attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_logical_toobig(ptr byval(%logical_toobig) align 1)
func.func private @takes_logical_toobig(%arg0: !ty_logical_toobig) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_logical_toobig(ptr sret(%logical_toobig) align 1)
func.func private @return_logical_toobig() -> !ty_logical_toobig attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_cmplx_toobig(ptr byval(%cmplx_toobig) align 4)
func.func private @takes_cmplx_toobig(%arg0: !ty_cmplx_toobig) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_cmplx_toobig(ptr sret(%cmplx_toobig) align 4)
func.func private @return_cmplx_toobig() -> !ty_cmplx_toobig attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_char_toobig(ptr byval(%char_toobig) align 1)
func.func private @takes_char_toobig(%arg0: !ty_char_toobig) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK: declare void @return_char_toobig(ptr sret(%char_toobig) align 1)
func.func private @return_char_toobig() -> !ty_char_toobig attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare { i32, float } @takes_and_return(float, float, float, float, float, float, float, float, ptr byval(%cmplx_f128) align 16)
func.func private @takes_and_return(%arg0: f32, %arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32,
                                    %arg5: f32, %arg6: f32, %arg7: f32, %arg8: !ty_cmplx_f128) -> !ty_i32_f32 attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_and_return2(ptr sret(%cmplx_f128) align 16, i32, i32, i32, i32, i32, i32, i32, i32, i64)
func.func private @takes_and_return2(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32,
                                     %arg5: i32, %arg6: i32, %arg7: i32, %arg8: !ty_i32_f32) -> !ty_cmplx_f128 attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}

// CHECK: declare void @takes_multi_byval_arguments(ptr byval(%cmplx_f128) align 16, ptr byval(%cmplx_f128) align 16, ptr byval(%cmplx_f128) align 16, ptr byval(%cmplx_f128) align 16, ptr byval(%cmplx_f128) align 16, ptr byval(%cmplx_f128) align 16, ptr byval(%cmplx_f128) align 16, i32, float, i64)
func.func private @takes_multi_byval_arguments(%arg0: !ty_cmplx_f128, %arg1: !ty_cmplx_f128, %arg2: !ty_cmplx_f128, %arg3: !ty_cmplx_f128,
                                               %arg4: !ty_cmplx_f128, %arg5: !ty_cmplx_f128, %arg6: !ty_cmplx_f128,
                                               %arg7: !ty_i32_f32, %arg8: !ty_i32_f32) attributes {fir.proc_attrs = #fir.proc_attrs<bind_c>}
}

