! Test diagnostics for unsupported statements between a loop-associated
! metadirective and its associated DO.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/threadprivate.f90 2>&1 | FileCheck --check-prefix=TP %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/requires.f90 2>&1 | FileCheck --check-prefix=REQ %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/prefetch.f90 2>&1 | FileCheck --check-prefix=PREFETCH %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/begin-prefetch.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=BEGIN-PREFETCH %s

! TP: A METADIRECTIVE in the specification part cannot have executable variants
! REQ: A METADIRECTIVE in the specification part cannot have executable variants
! PREFETCH: not yet implemented: PREFETCH compiler directive between
! PREFETCH-SAME: loop-associated METADIRECTIVE and its associated DO
! BEGIN-PREFETCH: not yet implemented: PREFETCH compiler directive between
! BEGIN-PREFETCH-SAME: loop-associated METADIRECTIVE and its associated DO

!--- threadprivate.f90
subroutine threadprivate_between(n, a)
  integer :: n, a(n), i
  real, save :: p
  !Invalid: metadirective with an executable variant in a specification part
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$omp threadprivate(p)  ! Must be in specification part
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- requires.f90
subroutine requires_between()
  integer :: a(10), i
  !Invalid: metadirective with an executable variant in a specification part
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$omp requires unified_address  ! Must be in specification part
  do i = 1, 10
    a(i) = i
  end do
end subroutine

!--- prefetch.f90
subroutine prefetch_between(a, n)
  integer :: a(n), n, i, idx
  logical :: choose
  external :: idx, choose
  !$omp metadirective &
  !$omp & when(user={condition(choose())}: do) &
  !$omp & otherwise(nothing)
  !dir$ prefetch a(idx())
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- begin-prefetch.f90
subroutine begin_prefetch_between(a, n)
  integer :: a(n), n, i, idx
  logical :: choose
  external :: idx, choose
  !$omp begin metadirective &
  !$omp & when(user={condition(choose())}: do) &
  !$omp & otherwise(nothing)
  !dir$ prefetch a(idx())
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine
