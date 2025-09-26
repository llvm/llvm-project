! RUN: %flang_fc1 -ffast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}
! REQUIRES: stability
! CHECK: module attributes {{{.*}}fir.fast_real_mod = true{{.*}}}

! CHECK-LABEL: @_QPmod_real4
subroutine mod_real4(r, a, p)
    implicit none
    real(kind=4) :: r, a, p
! CHECK: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<contract> : f32
! CHECK: %[[CV1:.*]] = fir.convert %[[DIV]] : (f32) -> si32
! CHECK: %[[CV2:.*]] = fir.convert %[[CV1]] : (si32) -> f32
! CHECK: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<contract> : f32
! CHECK: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<contract> : f32
! CHECK: fir.store %[[SUB]] to %[[R]] : !fir.ref<f32>
    r = mod(a, p)
end subroutine mod_real4

! CHECK-LABEL: @_QPmod_real8
subroutine mod_real8(r, a, p)
    implicit none
    real(kind=8) :: r, a, p
! CHECK: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<contract> : f64
! CHECK: %[[CV1:.*]] = fir.convert %[[DIV]] : (f64) -> si64
! CHECK: %[[CV2:.*]] = fir.convert %[[CV1]] : (si64) -> f64
! CHECK: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<contract> : f64
! CHECK: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<contract> : f64
! CHECK: fir.store %[[SUB]] to %[[R]] : !fir.ref<f64>
    r = mod(a, p)
end subroutine mod_real8

! CHECK-LABEL: @_QPmod_real10
subroutine mod_real10(r, a, p)
    implicit none
    real(kind=10) :: r, a, p
! CHECK-KIND10: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-KIND10: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-KIND10: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-KIND10: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-KIND10: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-KIND10: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<contract> : f80
! CHECK-KIND10: %[[CV1:.*]] = fir.convert %[[DIV]] : (f80) -> si80
! CHECK-KIND10: %[[CV2:.*]] = fir.convert %[[CV1]] : (si80) -> f80
! CHECK-KIND10: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<contract> : f80
! CHECK-KIND10: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<contract> : f80
! CHECK-KIND10: fir.store %[[SUB]] to %[[R]] : !fir.ref<f80>
    r = mod(a, p)
end subroutine mod_real10

! CHECK-LABEL: @_QPmod_real16
subroutine mod_real16(r, a, p)
    implicit none
    real(kind=16) :: r, a, p
! CHECK-KIND16: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-KIND16: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-KIND16: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-KIND16: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-KIND16: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-KIND16: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<contract> : f128
! CHECK-KIND16: %[[CV1:.*]] = fir.convert %[[DIV]] : (f128) -> si128
! CHECK-KIND16: %[[CV2:.*]] = fir.convert %[[CV1]] : (si128) -> f128
! CHECK-KIND16: %[[MUL:.*]] = arith.mulf %[[CV2]], %[[P_LOAD]] fastmath<contract> : f128
! CHECK-KIND16: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<contract> : f128
! CHECK-KIND16: fir.store %[[SUB]] to %[[R]] : !fir.ref<f128>
    r = mod(a, p)
end subroutine mod_real16
