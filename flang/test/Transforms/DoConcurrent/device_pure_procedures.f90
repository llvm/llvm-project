! Tests that, with `-fdo-concurrent-device-pure-procedures`, every pure
! procedure defined in the translation unit is implicitly `declare target` when
! `do concurrent` loops are mapped to the device, so that device-mapped loops in
! other translation units can call it. In that case, calls to procedures that
! are only declared here are assumed to have a device version and are not
! diagnosed.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device \
! RUN:   -fdo-concurrent-device-pure-procedures %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device \
! RUN:   -fdo-concurrent-to-openmp=device \
! RUN:   -fdo-concurrent-device-pure-procedures %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s \
! RUN:   -o - 2>&1 | FileCheck %s --check-prefix=NOFLAG
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host \
! RUN:   -fdo-concurrent-device-pure-procedures %s -o - 2>&1 \
! RUN:   | FileCheck %s --check-prefix=HOST

module m
  implicit none
  interface
    pure real function ext(x)
      real, intent(in) :: x
    end function
  end interface
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

  subroutine use_ext(a)
    real, intent(inout) :: a(:)
    integer :: i
    do concurrent (i = 1:size(a))
      a(i) = ext(a(i))
    end do
  end subroutine
end module

! CHECK-NOT: not marked DECLARE TARGET
! CHECK: func.func @_QMmPtwice({{.*}}) -> f32 attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}
! CHECK: func.func @_QMmPhalf({{.*}}) -> f32 attributes {{{.*}}omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}
! CHECK-NOT: omp.declare_target
! CHECK: func.func @_QMmPimpure_half(
! CHECK-NOT: omp.declare_target
! CHECK: func.func @_QMmPnot_pure(
! CHECK-NOT: omp.declare_target
! CHECK: func.func @_QMmPuse_ext(
! CHECK-NOT: omp.declare_target
! CHECK: func.func private @_QPext(
! CHECK-NOT: omp.declare_target

! NOFLAG: warning: {{.*}}'ext' is called from device code generated for a DO CONCURRENT loop
! NOFLAG-NOT: omp.declare_target

! HOST-NOT: omp.declare_target
