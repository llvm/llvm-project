!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

! An unsupported selector conservatively retains its OTHERWISE replacement.
subroutine f01(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
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
    !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
    !BECAUSE: COLLAPSE clause was specified with argument 2
    !$omp& otherwise(do collapse(2))
    do i = 1, n
      a(i) = i
    end do
  !$omp end metadirective
end subroutine
