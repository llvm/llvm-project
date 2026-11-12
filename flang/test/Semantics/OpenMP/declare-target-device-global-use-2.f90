! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 -J%t %t/declare-target-device-global-renaming.f90
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 -J%t %t/declare-target-device-use-renaming.f90 2>&1 | FileCheck %s

!--- declare-target-device-global-renaming.f90
module declare_target_device_global_renaming
  implicit none
  real :: marked
  real :: unmarked

  !$omp declare target(marked)
end module

!--- declare-target-device-use-renaming.f90
module declare_target_device_use_renaming
  use declare_target_device_global_renaming, only: renamed_marked => marked, renamed_unmarked => unmarked
  implicit none

contains
  subroutine uses_renamed_marked(x)
    !$omp declare target
    real :: x
    ! CHECK-NOT: warning: Variable 'marked' is referenced from an OpenMP DECLARE TARGET procedure but is not marked DECLARE TARGET [-Wopenmp-usage]
    x = renamed_marked
  end subroutine

  subroutine uses_renamed_unmarked(x)
    !$omp declare target
    real :: x
    ! CHECK: warning: Variable 'unmarked' is referenced from an OpenMP DECLARE TARGET procedure but is not marked DECLARE TARGET [-Wopenmp-usage]
    x = renamed_unmarked
  end subroutine

  subroutine host_only_uses_unmarked(x)
    !$omp declare target enter(host_only_uses_unmarked) device_type(host)
    real :: x
    ! CHECK-NOT: warning: Variable 'unmarked' is referenced from an OpenMP DECLARE TARGET procedure but is not marked DECLARE TARGET [-Wopenmp-usage]
    x = renamed_unmarked
  end subroutine
end module
