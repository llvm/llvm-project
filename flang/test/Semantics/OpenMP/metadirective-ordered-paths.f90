! RUN: not %flang_fc1 -fopenmp -fopenmp-version=51 -fsyntax-only %s 2>&1 | \
! RUN:   FileCheck %s --implicit-check-not=error:

! CHECK: error: Semantic errors in

! The outer choice and the inner selector produce PARALLEL -> DO first,
! then DO -> PARALLEL. Both paths have the same length and trait presence,
! but only the second matches the innermost ordered construct selector.
! Merging by trait presence alone would discard its COLLAPSE diagnostic.
subroutine ordered_paths(flag, n, a)
  logical :: flag
  integer :: n, a(n, n, n), i, j, k
  !$omp begin metadirective &
  !$omp& when(user={condition(flag)}: parallel) default(do)
  do i = 1, n
    !$omp begin metadirective &
    !$omp& when(construct={do}: parallel) default(do)
    do j = 1, n
      !$omp metadirective &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
      !$omp& when(construct={do, parallel}: simd collapse(2)) default(nothing)
      do k = 1, n
        a(k, j, i) = k
      end do
    end do
    !$omp end metadirective
  end do
  !$omp end metadirective
end subroutine

! PARALLEL -> DO alone must not match the reversed selector.
subroutine reversed_selector(n, a)
  integer :: n, a(n, n), i, j
  !$omp parallel do
  do i = 1, n
    !$omp metadirective &
    !$omp& when(construct={do, parallel}: simd collapse(2)) default(nothing)
    do j = 1, n
      a(j, i) = j
    end do
  end do
  !$omp end parallel do
end subroutine
