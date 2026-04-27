! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=50 %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_device_kind_host()
! CHECK:         omp.taskyield
! CHECK:         return
subroutine test_device_kind_host()
  !$omp metadirective &
  !$omp & when(device={kind(host)}: taskyield) &
  !$omp & default(nothing)
end subroutine

! CHECK-LABEL: func.func @_QPtest_multiple_when_second_match()
! CHECK-NOT:     omp.taskwait
! CHECK:         omp.taskyield
! CHECK:         return
subroutine test_multiple_when_second_match()
  !$omp metadirective &
  !$omp & when(implementation={vendor("amd")}: taskwait) &
  !$omp & when(device={kind(host)}: taskyield) &
  !$omp & default(nothing)
end subroutine
