! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

! Invalid capture statements in ATOMIC COMPARE CAPTURE.

subroutine f00(x, v, e, d)
  integer :: x, v, e, d
  !$omp atomic compare capture
  !ERROR: In ATOMIC UPDATE COMPARE CAPTURE the right-hand side of the capture assignment should read x
  v = v
  if (x == e) x = d
  !$omp end atomic
end

subroutine f01(x, v, e, d)
  integer :: x, v, e, d
  !$omp atomic compare capture
  if (x == e) x = d
  !ERROR: In ATOMIC UPDATE COMPARE CAPTURE the right-hand side of the capture assignment should read x
  v = e
  !$omp end atomic
end

subroutine f02(x, v, e, d)
  integer :: x, v, e, d
  !$omp atomic compare capture
  if (x == e) then
    x = d
  else
    !ERROR: In ATOMIC UPDATE COMPARE CAPTURE the right-hand side of the capture assignment should read x
    v = d
  end if
  !$omp end atomic
end

subroutine f03(x, e, d)
  integer :: x, e, d, v(2)
  !$omp atomic compare capture
  !ERROR: Capture variable v should be a scalar
  v = x
  if (x == e) x = d
  !$omp end atomic
end
