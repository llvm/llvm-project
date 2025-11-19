! RUN: %flang_fc1 -ffast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-FRM%if target=x86_64{{.*}} %{,CHECK-FRM-KIND10%}%if flang-supports-f128-math %{,CHECK-FRM-KIND16%}
! RUN: %flang_fc1 -ffast-math -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-FRM%if target=x86_64{{.*}} %{,CHECK-FRM-KIND10%}%if flang-supports-f128-math %{,CHECK-FRM-KIND16%}
! RUN: %flang_fc1 -ffast-math -fno-fast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-NFRM%if target=x86_64{{.*}} %{,CHECK-NFRM-KIND10%}%if flang-supports-f128-math %{,CHECK-NFRM-KIND16%}

! CHECK,CHECK-FRM: module attributes {{{.*}}fir.fast_real_mod = true{{.*}}}

! CHECK-LABEL: @_QPmod_real4
subroutine mod_real4(r, a, p)
    implicit none
    real(kind=4) :: r, a, p
! CHECK-FRM: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-FRM: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-FRM: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-FRM: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-FRM: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-FRM: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<{{.*}}> : f32
! CHECK-FRM: %[[CV1:.*]] = fir.convert %[[DIV]] : (f32) -> si32
! CHECK-FRM: %[[CV2:.*]] = fir.convert %[[CV1]] : (si32) -> f32
! CHECK-FRM: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<{{.*}}> : f32
! CHECK-FRM: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<{{.*}}> : f32
! CHECK-FRM: fir.store %[[SUB]] to %[[R]] : !fir.ref<f32>
! CHECK-NFRM: fir.call @_FortranAModReal4(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f32, f32, !fir.ref<i8>, i32) -> f32
    r = mod(a, p)
end subroutine mod_real4

! CHECK-LABEL: @_QPmod_real8
subroutine mod_real8(r, a, p)
    implicit none
    real(kind=8) :: r, a, p
! CHECK-FRM: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-FRM: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-FRM: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-FRM: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-FRM: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-FRM: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<{{.*}}> : f64
! CHECK-FRM: %[[CV1:.*]] = fir.convert %[[DIV]] : (f64) -> si64
! CHECK-FRM: %[[CV2:.*]] = fir.convert %[[CV1]] : (si64) -> f64
! CHECK-FRM: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<{{.*}}> : f64
! CHECK-FRM: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<{{.*}}> : f64
! CHECK-FRM: fir.store %[[SUB]] to %[[R]] : !fir.ref<f64>
! CHECK-NFRM: fir.call @_FortranAModReal8(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f64, f64, !fir.ref<i8>, i32) -> f64
    r = mod(a, p)
end subroutine mod_real8

! CHECK-LABEL: @_QPmod_real10
subroutine mod_real10(r, a, p)
    implicit none
    integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
    real(kind=kind10) :: r, a, p
! CHECK-FRM-KIND10: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-FRM-KIND10: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-FRM-KIND10: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-FRM-KIND10: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-FRM-KIND10: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-FRM-KIND10: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<{{.*}}> : f80
! CHECK-FRM-KIND10: %[[CV1:.*]] = fir.convert %[[DIV]] : (f80) -> si80
! CHECK-FRM-KIND10: %[[CV2:.*]] = fir.convert %[[CV1]] : (si80) -> f80
! CHECK-FRM-KIND10: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<{{.*}}> : f80
! CHECK-FRM-KIND10: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<{{.*}}> : f80
! CHECK-FRM-KIND10: fir.store %[[SUB]] to %[[R]] : !fir.ref<f80>
! CHECK-NFRM-KIND10: fir.call @_FortranAModReal10(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f80, f80, !fir.ref<i8>, i32) -> f80
    r = mod(a, p)
end subroutine mod_real10

! CHECK-LABEL: @_QPmod_real16
subroutine mod_real16(r, a, p)
    implicit none
    integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
    real(kind=kind16) :: r, a, p
! CHECK-FRM-KIND16: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-FRM-KIND16: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-FRM-KIND16: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-FRM-KIND16: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-FRM-KIND16: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-FRM-KIND16: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<{{.*}}> : f128
! CHECK-FRM-KIND16: %[[CV1:.*]] = fir.convert %[[DIV]] : (f128) -> si128
! CHECK-FRM-KIND16: %[[CV2:.*]] = fir.convert %[[CV1]] : (si128) -> f128
! CHECK-FRM-KIND16: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<{{.*}}> : f128
! CHECK-FRM-KIND16: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<{{.*}}> : f128
! CHECK-FRM-KIND16: fir.store %[[SUB]] to %[[R]] : !fir.ref<f128>
! CHECK-NFRM-KIND16: fir.call @_FortranAModReal16(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f128, f128, !fir.ref<i8>, i32) -> f128
    r = mod(a, p)
end subroutine mod_real16
