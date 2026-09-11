! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s

! TARGET hides the outer PARALLEL. The unreachable SIMD replacement must
! not be lowered, since its COLLAPSE clause is invalid for the single loop.
! CHECK-LABEL: func.func @_QPactual_target(
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK-NOT: omp.simd
! CHECK: return
subroutine actual_target(n, a)
  integer :: n, i, a(n)
  !$omp parallel
    !$omp target
      !$omp metadirective &
      !$omp& when(construct={parallel}: simd collapse(2)) default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end target
  !$omp end parallel
end subroutine

! A TARGET selected by a metadirective creates the same context boundary.
! CHECK-LABEL: func.func @_QPselected_target(
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK-NOT: omp.simd
! CHECK: return
subroutine selected_target(n, a)
  integer :: n, i, a(n)
  !$omp parallel
    !$omp begin metadirective default(target)
      !$omp metadirective &
      !$omp& when(construct={parallel, target}: simd collapse(2)) &
      !$omp& default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end metadirective
  !$omp end parallel
end subroutine

! The boundary includes TARGET itself and constructs nested inside it.
! CHECK-LABEL: func.func @_QPtarget_inner_parallel()
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK-NOT: omp.taskyield
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine target_inner_parallel()
  !$omp parallel
    !$omp target
      !$omp parallel
        !$omp metadirective when(construct={target, parallel}: barrier) &
        !$omp& default(taskyield)
      !$omp end parallel
    !$omp end target
  !$omp end parallel
end subroutine
