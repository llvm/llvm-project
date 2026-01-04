! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! EXPONENT

! CHECK-LABEL: exponent_test(
subroutine exponent_test(i1, i2, x4, x8)
    integer :: i1, i2, i3
    real(kind = 4) :: x4
    real(kind = 8) :: x8

    i1 = exponent(x4)
  ! CHECK: %[[temp0:.*]] = fir.load %{{.*}} : !fir.ref<f32>
  ! CHECK: fir.call @_FortranAExponent4_4(%[[temp0:.*]]) {{.*}}: (f32) -> i32

    i2 = exponent(x8)
  ! CHECK: %[[temp1:.*]] = fir.load %{{.*}} : !fir.ref<f64>
  ! CHECK: fir.call @_FortranAExponent8_4(%[[temp1:.*]]) {{.*}}: (f64) -> i32
end subroutine exponent_test

! CHECK-KIND10-LABEL: exponent_10(
subroutine exponent_10(i, x10)
    integer :: i
    integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
    real(kind = kind10) :: x10
    i = exponent(x10)
  ! CHECK-KIND10: %[[temp2:.*]] = fir.load %{{.*}} : !fir.ref<f80>
  ! CHECK-KIND10: fir.call @_FortranAExponent10_4(%[[temp2:.*]]) {{.*}}: (f80) -> i32
end subroutine

! CHECK-KIND16-LABEL: exponent_16(
subroutine exponent_16(i, x16)
    integer :: i
    integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
    real(kind = kind16) :: x16
    i = exponent(x16)
  ! CHECK-KIND16: %[[temp2:.*]] = fir.load %{{.*}} : !fir.ref<f128>
  ! CHECK-KIND16: fir.call @_FortranAExponent16_4(%[[temp2:.*]]) {{.*}}: (f128) -> i32
end subroutine
