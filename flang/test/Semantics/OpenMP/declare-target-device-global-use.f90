! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 -J%t %t/declare-target-device-globals.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 -J%t %t/declare-target-device-use.f90 2>&1 | FileCheck %s

!--- declare-target-device-globals.f90
module declare_target_device_globals
  implicit none
  real :: marked
  real :: unmarked
  real, parameter :: c = 1.0

  !$omp declare target(marked)
end module

!--- declare-target-device-use.f90
module declare_target_device_use
  use declare_target_device_globals
  implicit none

contains
  subroutine uses_marked(x)
    !$omp declare target
    real :: x
    x = marked
  end subroutine

  subroutine uses_unmarked(x)
    !$omp declare target
    real :: x
    real :: local
    local = x
    ! CHECK: warning: Variable 'unmarked' is referenced from an OpenMP DECLARE TARGET procedure but is not marked DECLARE TARGET [-Wopenmp-usage]
    ! CHECK-NOT: warning:
    x = unmarked + local + c
  end subroutine

  subroutine host_only_uses_unmarked(x)
    !$omp declare target enter(host_only_uses_unmarked) device_type(host)
    real :: x
    x = unmarked
  end subroutine
end module
