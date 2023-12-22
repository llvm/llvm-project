! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! 2.15.3 Data-Sharing Attribute Clauses
! 2.15.3.1 default Clause

subroutine default_none()
  integer a(3)
  integer, parameter :: D=10
  A = 1
  B = 2
  !$omp parallel default(none) private(c)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
  A(1:2) = 3
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
  B = 4
  C = 5 + D
  !$omp end parallel
end subroutine default_none

! Test that indices of sequential loops are privatised and hence do not error
! for DEFAULT(NONE)
subroutine default_none_seq_loop
  integer :: i

  !$omp parallel do default(none)
  do i = 1, 10
     do j = 1, 20
    enddo
  enddo
end subroutine

program mm
  call default_none()
  call default_none_seq_loop()
  !TODO: private, firstprivate, shared
end
