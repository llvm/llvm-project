! RUN: %flang_fc1 -ffast-real-mod -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-INLINE,CHECK-MT%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-INLINE-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-INLINE-KIND16%}
! RUN: %flang_fc1 -ffast-real-mod -fno-fast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-NFRM%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-NFRM-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-NFRM-KIND16%}
! RUN: %flang_fc1 -fno-fast-real-mod -ffast-real-mod -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-INLINE,CHECK-MT%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-INLINE-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-INLINE-KIND16%}
! RUN: %flang_fc1 -ffast-math -emit-mlir -o - %s | FileCheck %s -DFMF=fast --check-prefixes=CHECK,CHECK-INLINE,CHECK-MT%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-INLINE-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-INLINE-KIND16%}
! RUN: %flang_fc1 -ffast-math -fno-fast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-NFRM%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-NFRM-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-NFRM-KIND16%}
! RUN: %flang_fc1 -ffast-real-mod -mllvm -fast-real-mod-trunc=math-trunc -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-INLINE,CHECK-MT%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-INLINE-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-INLINE-KIND16%}
! RUN: %flang_fc1 -ffast-real-mod -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-INLINE,CHECK-EMU%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-INLINE-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-INLINE-KIND16%}
! RUN: %flang_fc1 -ffast-math -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s -DFMF=fast --check-prefixes=CHECK,CHECK-INLINE,CHECK-EMU%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-INLINE-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-INLINE-KIND16%}
! RUN: %flang_fc1 -ffast-math -fno-fast-real-mod -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-NFRM%if target=x86_64{{.*}} %{,CHECK-KIND10,CHECK-NFRM-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16,CHECK-NFRM-KIND16%}

! CHECK-INLINE: module attributes {{{.*}}fir.fast_real_mod = true{{.*}}}

! CHECK-LABEL: @_QPmod_real4
subroutine mod_real4(r, a, p)
    implicit none
    real(kind=4) :: r, a, p
! CHECK-EMU: %[[BOUND:.*]] = arith.constant 0x4B000000 : f32
! CHECK: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-INLINE: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<[[FMF]]> : f32
! CHECK-MT: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<[[FMF]]> : f32
! CHECK-EMU: %[[ABS:.*]] = math.absf %[[DIV]] fastmath<[[FMF]]> : f32
! CHECK-EMU: %[[CMP:.*]] = arith.cmpf olt, %[[ABS]], %[[BOUND]] fastmath<[[FMF]]> : f32
! CHECK-EMU: %[[INT:.*]] = arith.fptosi %[[DIV]] : f32 to i32
! CHECK-EMU: %[[FLT:.*]] = arith.sitofp %[[INT]] : i32 to f32
! CHECK-EMU: %[[TRUNC:.*]] = arith.select %[[CMP]], %[[FLT]], %[[DIV]] : f32
! CHECK-INLINE: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<[[FMF]]> : f32
! CHECK-INLINE: %[[RES:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<[[FMF]]> : f32
! CHECK-NFRM: %[[RES:.*]] = fir.call @_FortranAModReal4(%[[A_LOAD]], %[[P_LOAD]], %{{.*}}, %{{.*}}) {{.*}}: (f32, f32, !fir.ref<i8>, i32) -> f32
! CHECK: fir.store %[[RES]] to %[[R]] : !fir.ref<f32>
    r = mod(a, p)
end subroutine mod_real4

! CHECK-LABEL: @_QPmod_real8
subroutine mod_real8(r, a, p)
    implicit none
    real(kind=8) :: r, a, p
! CHECK-EMU: %[[BOUND:.*]] = arith.constant 0x4330000000000000 : f64
! CHECK: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-INLINE: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<[[FMF]]> : f64
! CHECK-MT: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<[[FMF]]> : f64
! CHECK-EMU: %[[ABS:.*]] = math.absf %[[DIV]] fastmath<[[FMF]]> : f64
! CHECK-EMU: %[[CMP:.*]] = arith.cmpf olt, %[[ABS]], %[[BOUND]] fastmath<[[FMF]]> : f64
! CHECK-EMU: %[[INT:.*]] = arith.fptosi %[[DIV]] : f64 to i64
! CHECK-EMU: %[[FLT:.*]] = arith.sitofp %[[INT]] : i64 to f64
! CHECK-EMU: %[[TRUNC:.*]] = arith.select %[[CMP]], %[[FLT]], %[[DIV]] : f64
! CHECK-INLINE: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<[[FMF]]> : f64
! CHECK-INLINE: %[[RES:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<[[FMF]]> : f64
! CHECK-NFRM: %[[RES:.*]] = fir.call @_FortranAModReal8(%[[A_LOAD]], %[[P_LOAD]], %{{.*}}, %{{.*}}) {{.*}}: (f64, f64, !fir.ref<i8>, i32) -> f64
! CHECK: fir.store %[[RES]] to %[[R]] : !fir.ref<f64>
    r = mod(a, p)
end subroutine mod_real8

! REAL(10) and REAL(16) use math.trunc with -fast-real-mod-trunc=emulate-trunc
! too, because their integer conversions are library calls.

! CHECK-LABEL: @_QPmod_real10
subroutine mod_real10(r, a, p)
    implicit none
    integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
    real(kind=kind10) :: r, a, p
! CHECK-KIND10: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-KIND10: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-KIND10: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-KIND10: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-KIND10: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-INLINE-KIND10: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<[[FMF]]> : f80
! CHECK-INLINE-KIND10: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<[[FMF]]> : f80
! CHECK-INLINE-KIND10: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<[[FMF]]> : f80
! CHECK-INLINE-KIND10: %[[RES:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<[[FMF]]> : f80
! CHECK-NFRM-KIND10: %[[RES:.*]] = fir.call @_FortranAModReal10(%[[A_LOAD]], %[[P_LOAD]], %{{.*}}, %{{.*}}) {{.*}}: (f80, f80, !fir.ref<i8>, i32) -> f80
! CHECK-KIND10: fir.store %[[RES]] to %[[R]] : !fir.ref<f80>
    r = mod(a, p)
end subroutine mod_real10

! CHECK-LABEL: @_QPmod_real16
subroutine mod_real16(r, a, p)
    implicit none
    integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
    real(kind=kind16) :: r, a, p
! CHECK-KIND16: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-KIND16: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-KIND16: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-KIND16: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-KIND16: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-INLINE-KIND16: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<[[FMF]]> : f128
! CHECK-INLINE-KIND16: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<[[FMF]]> : f128
! CHECK-INLINE-KIND16: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<[[FMF]]> : f128
! CHECK-INLINE-KIND16: %[[RES:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<[[FMF]]> : f128
! CHECK-NFRM-KIND16: %[[RES:.*]] = fir.call @_FortranAModReal16(%[[A_LOAD]], %[[P_LOAD]], %{{.*}}, %{{.*}}) {{.*}}: (f128, f128, !fir.ref<i8>, i32) -> f128
! CHECK-KIND16: fir.store %[[RES]] to %[[R]] : !fir.ref<f128>
    r = mod(a, p)
end subroutine mod_real16
