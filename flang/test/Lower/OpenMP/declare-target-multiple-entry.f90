!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

! Test that a bare '!$omp declare target' inside a multiple entry subprogram
! marks all entries as declare-target functions.

! CHECK: func.func @_QPfirst_entry{{.*}}device_type = (any)
! CHECK: func.func @_QPsecond_entry{{.*}}device_type = (any)
! CHECK: func.func @_QPthird_entry{{.*}}device_type = (any)

subroutine first_entry()
  implicit none
  !$omp declare target

  call foo()
  return

entry second_entry()
  call bar()
  return

entry third_entry()
  call baz()
  return
end subroutine
