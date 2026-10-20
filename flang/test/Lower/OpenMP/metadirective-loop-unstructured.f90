! A DO associated with a METADIRECTIVE variant whose body branches only within
! itself. The computed GO TO and its targets are all inside the loop body, so
! the loop keeps its structured form and the raw blocks are confined to an
! scf.execute_region inside omp.loop_nest. This used to be rejected as not yet
! implemented, because the body's blocks had nowhere to live.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/static.f90 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/runtime.f90 | FileCheck %s

! CHECK:   omp.wsloop
! CHECK:     omp.loop_nest
! CHECK:       scf.execute_region no_inline {
! The computed GO TO and the branches it feeds stay inside the region.
! CHECK:         fir.select %{{[0-9]+}} : i32 [1, ^bb[[L10:[0-9]+]], 2, ^bb[[L20:[0-9]+]], unit, ^bb[[L10]]]
! CHECK:       ^bb[[L10]]:
! CHECK:       ^bb[[L20]]:
! CHECK:         scf.yield
! CHECK:       omp.yield

!--- static.f90
subroutine test_static(n, a, selector)
  integer :: n, a(n), selector, i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    go to (10, 20), selector
10  a(i) = 1
    go to 30
20  a(i) = 2
30  continue
  end do
end subroutine

!--- runtime.f90
subroutine test_runtime(flag, n, a, selector)
  logical :: flag
  integer :: n, a(n), selector, i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    go to (10, 20), selector
10  a(i) = 1
    go to 30
20  a(i) = 2
30  continue
  end do
end subroutine
