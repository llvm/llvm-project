! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! FRACTION

! CHECK-LABEL: fraction_test(
subroutine fraction_test(res4, res8, x4, x8)
    real(kind = 4) :: x4, res4
    real(kind = 8) :: x8, res8

    res4 = fraction(x4)
  ! CHECK: %[[temp0:.*]] = fir.load %{{.*}} : !fir.ref<f32>
  ! CHECK: fir.call @_FortranAFraction4(%[[temp0:.*]]) {{.*}}: (f32) -> f32

    res8 = fraction(x8)
  ! CHECK: %[[temp1:.*]] = fir.load %{{.*}} : !fir.ref<f64>
  ! CHECK: fir.call @_FortranAFraction8(%[[temp1:.*]]) {{.*}}: (f64) -> f64
end subroutine fraction_test

! CHECK-KIND10-LABEL: fraction_10(
subroutine fraction_10(res10, x10)
    integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
    real(kind = kind10) :: x10, res10
    res10 = fraction(x10)
  ! CHECK-KIND10: %[[temp2:.*]] = fir.load %{{.*}} : !fir.ref<f80>
  ! CHECK-KIND10: fir.call @_FortranAFraction10(%[[temp2:.*]]) {{.*}}: (f80) -> f80
end subroutine

! CHECK-KIND16-LABEL: fraction_16(
subroutine fraction_16(res16, x16)
    integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
    real(kind = kind16) :: x16, res16
    res16 = fraction(x16)
  ! CHECK-KIND16: %[[temp2:.*]] = fir.load %{{.*}} : !fir.ref<f128>
  ! CHECK-KIND16: fir.call @_FortranAFraction16(%[[temp2:.*]]) {{.*}}: (f128) -> f128
end subroutine
