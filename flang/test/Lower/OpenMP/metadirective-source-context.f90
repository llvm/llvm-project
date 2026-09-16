! RUN: %if x86-registered-target %{ %flang_fc1 -fopenmp \
! RUN:   -fopenmp-version=52 -triple x86_64-unknown-linux-gnu \
! RUN:   -emit-hlfir %s -o - | FileCheck %s %}

! Executable loop transformations contribute to the construct context.
! CHECK-LABEL: func.func @_QPtile_context(
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine tile_context(n, a)
  integer :: n, a(n, n), i, j
  !$omp tile sizes(2)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: taskyield) &
    !$omp& when(device={arch(x86_64)}: barrier)
    do j = 1, n
      a(j, i) = j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunroll_context(
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine unroll_context(n, a)
  integer :: n, a(n), i
  !$omp unroll partial(2)
  do i = 1, n
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: taskyield) &
    !$omp& when(device={arch(x86_64)}: barrier)
    a(i) = i
  end do
end subroutine

! Informational directives do not contribute to the construct context.
! CHECK-LABEL: func.func @_QPassume_context()
! CHECK: omp.taskyield
! CHECK-NOT: omp.barrier
! CHECK: return
subroutine assume_context()
  !$omp assume holds(.true.)
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: taskyield) &
    !$omp& when(device={arch(x86_64)}: barrier)
  !$omp end assume
end subroutine

! Combined and composite constructs retain their source nesting order.
! CHECK-LABEL: func.func @_QPcombined_context(
! CHECK: omp.teams
! CHECK: omp.parallel
! CHECK: omp.distribute
! CHECK: omp.wsloop
! CHECK: omp.taskyield
! CHECK-NOT: omp.barrier
! CHECK: return
subroutine combined_context(n)
  integer :: n, i
  !$omp teams distribute parallel do
  do i = 1, n
    !$omp metadirective &
    !$omp& when(implementation={vendor(score(3): llvm)}: barrier) &
    !$omp& when(construct={parallel}: taskyield)
  end do
end subroutine
