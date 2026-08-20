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

! CHECK-LABEL: func.func @_QPtest_begin_device_kind_host()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK:         return
subroutine test_begin_device_kind_host()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(device={kind(host)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_device_kind_nohost()
! CHECK-NOT:     omp.parallel
! CHECK:         return
subroutine test_begin_device_kind_nohost()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(device={kind(nohost)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_multiple_when()
! CHECK:         omp.parallel
! CHECK:           omp.terminator
! CHECK-NOT:     omp.task
! CHECK:         return
subroutine test_begin_multiple_when()
  integer :: x
  x = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor("amd")}: task) &
  !$omp & when(device={kind(host)}: parallel)
  x = 1
  !$omp end metadirective
end subroutine
