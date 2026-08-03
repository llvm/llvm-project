! A selectable loop variant needs a variant-local loop-IV binding before it can
! be nested in another OpenMP data environment.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/static.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/dynamic.f90 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %t/selected-metadirective.f90 2>&1 | FileCheck --check-prefix=SELECTED %s

! CHECK: not yet implemented: loop-associated METADIRECTIVE nested in an OpenMP data environment
! SELECTED: not yet implemented: data-environment construct in METADIRECTIVE variant

!--- static.f90
subroutine test_static_do_in_parallel(n, a, after)
  integer :: n, a(n), after, i
  i = 0
  !$omp parallel num_threads(1) shared(n, a, after, i)
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end parallel
end subroutine

!--- dynamic.f90
subroutine test_do_in_parallel(flag, n, a, after)
  logical, intent(in) :: flag
  integer :: n, a(n), after, i
  i = 0
  !$omp parallel num_threads(1) shared(flag, n, a, after, i)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end parallel
end subroutine

!--- selected-metadirective.f90
subroutine test_do_in_selected_parallel(flag, n, a, after)
  logical, intent(in) :: flag
  integer :: n, a(n), after, i
  i = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: &
  !$omp &   parallel num_threads(1) shared(flag, n, a, after, i)) &
  !$omp & otherwise(nothing)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end metadirective
end subroutine
