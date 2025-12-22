! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! CHECK-LABEL: func @_QPmod_testr4(
subroutine mod_testr4(r, a, p)
  real(4) :: r, a, p
! CHECK: %[[LINE:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK: %[[FILE:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK: %[[FILEARG:.*]] = fir.convert %[[FILE]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAModReal4(%[[A_LOAD]], %[[P_LOAD]], %[[FILEARG]], %[[LINE]]) {{.*}}: (f32, f32, !fir.ref<i8>, i32) -> f32
  r = mod(a, p)
end subroutine

! CHECK-LABEL: func @_QPmod_testr8(
subroutine mod_testr8(r, a, p)
  real(8) :: r, a, p
! CHECK: fir.call @_FortranAModReal8(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f64, f64, !fir.ref<i8>, i32) -> f64
  r = mod(a, p)
end subroutine

! CHECK-KIND10-LABEL: func @_QPmod_testr10(
subroutine mod_testr10(r, a, p)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind10) :: r, a, p
! CHECK-KIND10: fir.call @_FortranAModReal10(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f80, f80, !fir.ref<i8>, i32) -> f80
  r = mod(a, p)
end subroutine

! CHECK-KIND16-LABEL: func @_QPmod_testr16(
subroutine mod_testr16(r, a, p)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind16) :: r, a, p
! CHECK-KIND16: fir.call @_FortranAModReal16(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (f128, f128, !fir.ref<i8>, i32) -> f128
  r = mod(a, p)
end subroutine
