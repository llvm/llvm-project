! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! SET_EXPONENT
! CHECK-LABEL: set_exponent_test_4
subroutine set_exponent_test_4(x, i)
  real(kind = 4) :: x
  integer :: i
  x = set_exponent(x, i)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}"_QFset_exponent_test_4Ei"
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}"_QFset_exponent_test_4Ex"
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranASetExponent4(%[[VAL_5]], %[[VAL_7]]) fastmath<contract> : (f32, i64) -> f32
! CHECK:  hlfir.assign %[[VAL_8]] to %[[VAL_4]]#0 : f32, !fir.ref<f32>
end subroutine


! CHECK-LABEL: set_exponent_test_8
subroutine set_exponent_test_8(x, i)
  real(kind = 8) :: x
  integer :: i
  x = set_exponent(x, i)
! CHECK: fir.call @_FortranASetExponent8(%{{.*}}, %{{.*}}) {{.*}}: (f64, i64) -> f64
end subroutine

! CHECK-KIND10-LABEL: set_exponent_test_10
subroutine set_exponent_test_10(x, i)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(kind = kind10) :: x
  integer :: i
  x = set_exponent(x, i)
! CHECK-KIND10: fir.call @_FortranASetExponent10(%{{.*}}, %{{.*}}) {{.*}}: (f80, i64) -> f80
end subroutine

! CHECK-KIND16-LABEL: set_exponent_test_16
subroutine set_exponent_test_16(x, i)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(kind = kind16) :: x
  integer :: i
  x = set_exponent(x, i)
! CHECK-KIND16: fir.call @_FortranASetExponent16(%{{.*}}, %{{.*}}) {{.*}}: (f128, i64) -> f128
end subroutine
