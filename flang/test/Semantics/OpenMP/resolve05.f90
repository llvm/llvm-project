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

! I/O implied-DO variables are predetermined private in the innermost
! enclosing parallel construct. See https://github.com/llvm/llvm-project/issues/197396.
subroutine default_none_output_implied_do
  integer :: ido

  call omp_set_num_threads(2)
  !$omp parallel default(none)
  print *, (ido, ido=1,2)
  !$omp end parallel
  print *, 'pass'
end subroutine

subroutine default_none_output_implied_do_arrays
  use omp_lib
  implicit none
  integer, parameter :: n = 10
  integer :: a(n) = 0, i = 0, b(n) = 0

  !$omp parallel default(none) private(a) shared(b)
  a = omp_get_thread_num()
  !$omp critical
  b = b + a
  print *, (b(i), a(i), i=2,4)
  !$omp end critical
  !$omp end parallel
end subroutine

subroutine default_none_input_implied_do
  implicit none
  integer, parameter :: n = 10
  integer :: a(n), i

  !$omp parallel default(none) shared(a)
  read *, (a(i), i=1,n)
  !$omp end parallel
end subroutine

subroutine default_none_simd_loop(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  !$omp parallel default(none) shared(n)
  !$omp simd
  do i = 1, n
  end do
  !$omp end simd
  !$omp end parallel
end subroutine

subroutine default_none_simd_loop_outer_reference(n)
  implicit none
  integer, intent(in) :: n
  integer :: i

  !$omp parallel default(none) shared(n)
  !$omp simd
  do i = 1, n
  end do
  !$omp end simd
  !ERROR: The DEFAULT(NONE) clause requires that 'i' must be listed in a data-sharing attribute clause
  print *, i
  !$omp end parallel
end subroutine

! Test that DEFAULT(NONE) error check sees implicit references
subroutine default_none_nested()
  integer :: a

  !$omp parallel default(none)
  !$omp task
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
  a = 1
  !$omp end task
  !$omp end parallel
end subroutine default_none_nested

! Test that we do not error for an explicitly privatized variable
subroutine default_none_private()
  integer :: a

  !$omp parallel default(none)
  !$omp task private(a)
    a = 1
  !$omp end task
  !$omp end parallel
end subroutine

program mm
  call default_none()
  call default_none_seq_loop()
  call default_none_nested()
  call default_none_private()
  !TODO: private, firstprivate, shared
end
