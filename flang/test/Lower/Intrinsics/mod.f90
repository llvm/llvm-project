! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}
! RUN: %flang_fc1 -emit-hlfir -fcheck-integer-mod-zero %s -o - | FileCheck %s --check-prefix=CHECK-MOD-ZERO

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

! By default, integer MOD remains an unchecked inlined remainder.
! CHECK-LABEL: func @_QPmod_testi4(
! CHECK-NOT: fir.call @_FortranAReportFatalUserError
! CHECK: arith.remsi %{{.*}}, %{{.*}} : i32

! With -fcheck-integer-mod-zero, a divisor that is not a known nonzero
! constant is tested and a fatal error is reported.
! CHECK-MOD-ZERO-LABEL: func @_QPmod_testi4(
subroutine mod_testi4(r, a, p)
  integer(4) :: r, a, p
! CHECK-MOD-ZERO: %[[A:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-MOD-ZERO: %[[P:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK-MOD-ZERO: %[[ISZERO:.*]] = arith.cmpi eq, %[[P]], %c0{{.*}} : i32
! CHECK-MOD-ZERO: fir.if %[[ISZERO]] {
! CHECK-MOD-ZERO:   fir.call @_FortranAReportFatalUserError
! CHECK-MOD-ZERO: }
! CHECK-MOD-ZERO: arith.remsi %[[A]], %[[P]] : i32
  r = mod(a, p)
end subroutine

! CHECK-MOD-ZERO-LABEL: func @_QPmod_testi8(
subroutine mod_testi8(r, a, p)
  integer(8) :: r, a, p
! CHECK-MOD-ZERO: fir.call @_FortranAReportFatalUserError
! CHECK-MOD-ZERO: arith.remsi %{{.*}}, %{{.*}} : i64
  r = mod(a, p)
end subroutine

! A constant nonzero divisor keeps the inlined remainder with no test.
! CHECK-MOD-ZERO-LABEL: func @_QPmod_testi4_constant(
subroutine mod_testi4_constant(r, a)
  integer(4) :: r, a
! CHECK-MOD-ZERO-NOT: fir.call @_FortranAReportFatalUserError
! CHECK-MOD-ZERO: arith.remsi %{{.*}}, %c8{{.*}} : i32
  r = mod(a, 8)
end subroutine
