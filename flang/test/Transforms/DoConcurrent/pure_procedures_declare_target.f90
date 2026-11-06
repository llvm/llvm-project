! Tests that, when `do concurrent` loops are mapped to the device, every pure
! procedure defined in the translation unit is implicitly `declare target`, so
! that device-mapped loops in other translation units can call it.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device \
! RUN:   -fdo-concurrent-to-openmp=device %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s --check-prefix=HOST

module m
  implicit none
contains
  pure real function twice(x)
    real, intent(in) :: x
    twice = 2 * x
  end function

  elemental real function half(x)
    real, intent(in) :: x
    half = x / 2
  end function

  impure elemental real function impure_half(x)
    real, intent(in) :: x
    impure_half = x / 2
  end function

  real function not_pure(x)
    real, intent(in) :: x
    not_pure = x
  end function
end module

! CHECK: func.func @_QMmPtwice({{.*}}) -> f32 attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}
! CHECK: func.func @_QMmPhalf({{.*}}) -> f32 attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}
! CHECK-NOT: omp.declare_target
! CHECK: func.func @_QMmPimpure_half(
! CHECK-NOT: omp.declare_target
! CHECK: func.func @_QMmPnot_pure(
! CHECK-NOT: omp.declare_target

! HOST-NOT: omp.declare_target
