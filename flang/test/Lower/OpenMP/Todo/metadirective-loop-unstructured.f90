! Defer unstructured associated loops until every selection path can give its
! PFT blocks an independent mapping.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/static.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/runtime.f90 2>&1 | FileCheck %s

! CHECK: not yet implemented: unstructured associated DO in loop-associated METADIRECTIVE variant

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
