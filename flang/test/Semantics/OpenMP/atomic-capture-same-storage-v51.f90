! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

! v and x can't be the same variable in ATOMIC READ/CAPTURE. Since OpenMP
! 5.2, v and e also can't be the same when v gets the original value of x.
! Other overlaps are still warnings.

subroutine f00(x)
  integer :: x
  !$omp atomic read
  !ERROR: Within atomic operation x and x access the same storage
  x = x
end

subroutine f01(x)
  integer :: x
  !$omp atomic capture
  !ERROR: Within atomic operation x and x access the same storage
  x = x
  x = x + 1
  !$omp end atomic
end

subroutine f02(x, e, d)
  integer :: x, e, d
  !$omp atomic compare capture
  !ERROR: Within atomic operation x and x access the same storage
  x = x
  if (x == e) x = d
  !$omp end atomic
end

subroutine f03(x)
  integer :: x
  !$omp atomic write
  !WARNING: Within atomic operation x and x+1_4 access the same storage
  x = x + 1
end

! v gets the original value of x.
subroutine f04(x, v, d)
  integer :: x, v, d
  !$omp atomic compare capture
  v = x
  if (x == v) x = d
  !$omp end atomic
end

! v gets the final value of x, so v and e can be the same.
subroutine f05(x, v, d)
  integer :: x, v, d
  !$omp atomic compare capture
  if (x == v) x = d
  v = x
  !$omp end atomic
end
