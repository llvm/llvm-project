! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 -o - %t/collapse.f90 2>&1 | FileCheck %s --check-prefix=COLLAPSE
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 -o - %t/negative-step.f90 2>&1 | FileCheck %s --check-prefix=NEGATIVE-STEP
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 -o - %t/dynamic-step.f90 2>&1 | FileCheck %s --check-prefix=DYNAMIC-STEP
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 -o - %t/negative-lb.f90 2>&1 | FileCheck %s --check-prefix=NEGATIVE-LB

! Device lowering only supports a single forward loop with a constant
! non-negative lower bound and constant positive step; other shapes emit a TODO.

! COLLAPSE: not yet implemented: lastprivate(conditional:) with collapse on a target region
! NEGATIVE-STEP: not yet implemented: lastprivate(conditional:) with a non-positive or non-constant step on a target region
! DYNAMIC-STEP: not yet implemented: lastprivate(conditional:) with a non-positive or non-constant step on a target region
! NEGATIVE-LB: not yet implemented: lastprivate(conditional:) with a negative or non-constant lower bound on a target region

!--- collapse.f90
subroutine collapse_case(n, m, a, x)
  integer :: n, m, i, j, a(n, m), x
  x = 0
  !$omp target parallel do collapse(2) map(tofrom: x) map(to: a) lastprivate(conditional: x)
  do i = 1, n
    do j = 1, m
      if (a(i, j) > 0) x = a(i, j)
    end do
  end do
end subroutine

!--- negative-step.f90
subroutine negative_step_case(n, a, x)
  integer :: n, i, a(n), x
  x = 0
  !$omp target parallel do map(tofrom: x) map(to: a) lastprivate(conditional: x)
  do i = n, 1, -1
    if (a(i) > 0) x = a(i)
  end do
end subroutine

!--- dynamic-step.f90
subroutine dynamic_step_case(n, k, a, x)
  integer :: n, k, i, a(n), x
  x = 0
  !$omp target parallel do map(tofrom: x) map(to: a) lastprivate(conditional: x)
  do i = 1, n, k
    if (a(i) > 0) x = a(i)
  end do
end subroutine

!--- negative-lb.f90
subroutine negative_lb_case(a, x)
  integer :: i, a(-10:-1), x
  x = 0
  !$omp target parallel do map(tofrom: x) map(to: a) lastprivate(conditional: x)
  do i = -10, -1
    if (a(i) > 0) x = a(i)
  end do
end subroutine
