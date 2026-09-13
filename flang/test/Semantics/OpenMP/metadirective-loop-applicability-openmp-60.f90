! RUN: not %flang_fc1 -fopenmp -fopenmp-version=60 -fsyntax-only %s 2>&1 | \
! RUN:   FileCheck %s --implicit-check-not=error:

! CHECK: error: Semantic errors in

! An unsupported selector conservatively retains its OTHERWISE replacement.
subroutine f01(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
  !$omp& when(target_device={kind(host)}: nothing) otherwise(do collapse(2))
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Unsupported-selector recovery retains an implicit NOTHING fallback.
subroutine f02(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp& when(construct={simd(simdlen(8))}: parallel)
    !$omp metadirective &
    !$omp& when(construct={parallel}: nothing) &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a nest of depth 2, but the associated nest is a nest of depth 1
! CHECK: because: COLLAPSE clause was specified with argument 2
    !$omp& otherwise(do collapse(2))
    do i = 1, n
      a(i) = i
    end do
  !$omp end metadirective
end subroutine

! A loop transformation in APPLY inherits the reachability of the
! metadirective replacement that contains it.
subroutine unreachable_transformation()
  !$omp metadirective &
  !$omp& when(user={condition(score(10): .true.)}: nothing) &
  !$omp& when(user={condition(score(5): .true.)}: &
  !$omp& tile sizes(2) apply(grid: unroll)) &
  !$omp& otherwise(nothing)
end subroutine

subroutine reachable_transformation(flag)
  logical :: flag
  !$omp metadirective &
  !$omp& when(user={condition(score(10): flag)}: nothing) &
  !$omp& when(user={condition(score(5): .true.)}: &
! CHECK: :[[@LINE+4]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
  !$omp& tile sizes(2) apply(grid: unroll)) &
  !$omp& otherwise(nothing)
end subroutine

! NOTHING with APPLY must retain its specification when selected.
subroutine selected_apply()
  !$omp metadirective &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
  !$omp& when(user={condition(.true.)}: nothing apply(reverse)) &
  !$omp& otherwise(nothing)
end subroutine

! A selected fallback must also retain its APPLY specifications.
subroutine fallback_apply()
  !$omp metadirective when(user={condition(.false.)}: nothing) &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
  !$omp& otherwise(nothing apply(reverse))
end subroutine

! A dynamic condition leaves the APPLY specification reachable.
subroutine dynamic_apply(flag)
  logical :: flag
  !$omp metadirective &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
  !$omp& when(user={condition(flag)}: nothing apply(reverse)) &
  !$omp& otherwise(nothing)
end subroutine

! Unsupported-selector recovery must preserve NOTHING with APPLY as well.
subroutine unsupported_apply()
  !$omp metadirective when(target_device={kind(host)}: nothing) &
! CHECK: :[[@LINE+2]]:{{[0-9]+}}: error: This construct should contain
! CHECK-SAME: a DO-loop or a loop-nest-generating construct
  !$omp& otherwise(nothing apply(reverse))
end subroutine

! The applied transformation must be checked against an associated loop.
subroutine insufficient_depth(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
! CHECK: :[[@LINE+3]]:{{[0-9]+}}: error: This construct requires
! CHECK-SAME: a perfect nest of depth 2
! CHECK-SAME: but the associated nest is a perfect nest of depth 1
  !$omp& otherwise(nothing apply(tile sizes(2, 2)))
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Statically false and lower-ranked replacements must remain unchecked.
subroutine unreachable_apply()
  !$omp metadirective &
  !$omp& when(user={condition(.false.)}: nothing apply(reverse)) &
  !$omp& otherwise(nothing)
end subroutine

subroutine lower_ranked_apply()
  !$omp metadirective &
  !$omp& when(user={condition(score(1): .true.)}: nothing) &
  !$omp& when(user={condition(.true.)}: nothing apply(reverse)) &
  !$omp& otherwise(nothing apply(reverse))
end subroutine

! A reachable APPLY with a suitable associated loop is valid.
subroutine valid_apply(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp& when(user={condition(.true.)}: nothing apply(reverse)) &
  !$omp& otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine
