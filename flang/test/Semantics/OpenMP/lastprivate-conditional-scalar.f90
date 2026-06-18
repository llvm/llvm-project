! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! [5.2:108:5-7] A list item that appears in a lastprivate clause with a
! conditional modifier must be a scalar variable.

subroutine foo()
  integer :: s, i
  integer :: arr(10)
  integer :: mat(3, 3)

  ! Scalar list item is allowed.
  !$omp do lastprivate(conditional: s)
  do i = 1, 100
    if (mod(i, 2) == 0) s = i
  enddo
  !$omp end do

!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable, 'arr' is not scalar
  !$omp do lastprivate(conditional: arr)
  do i = 1, 100
    if (mod(i, 2) == 0) arr(1) = i
  enddo
  !$omp end do

!ERROR: A list item that appears in a LASTPRIVATE clause with the CONDITIONAL modifier must be a scalar variable, 'mat' is not scalar
  !$omp do lastprivate(conditional: mat)
  do i = 1, 100
    if (mod(i, 2) == 0) mat(1, 1) = i
  enddo
  !$omp end do
end
