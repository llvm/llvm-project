!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

! A LASTPRIVATE clause with the CONDITIONAL modifier must name a whole scalar
! variable.  Aggregate designators -- array elements, array sections, and
! structure components -- are rejected by the general LASTPRIVATE object
! diagnostics; verify they fail gracefully (no silent acceptance) when the
! conditional modifier is present.

subroutine aggregates(n)
  integer :: n, i
  integer :: arr(10)
  type t
    integer :: c
  end type
  type(t) :: dt

!ERROR: An array element cannot appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(conditional: arr(1))
  do i = 1, n
    arr(1) = i
  end do
  !$omp end parallel do

!ERROR: An array element cannot appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(conditional: arr(1:5))
  do i = 1, n
    arr(1) = i
  end do
  !$omp end parallel do

!ERROR: A structure component cannot appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(conditional: dt%c)
  do i = 1, n
    dt%c = i
  end do
  !$omp end parallel do
end subroutine
