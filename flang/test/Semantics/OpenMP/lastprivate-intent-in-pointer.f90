! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.15.3.5 lastprivate Clause
! Pointers with the INTENT(IN) attribute may not appear in a lastprivate clause.

subroutine omp_lastprivate(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(p)
  do i = 1, 10
    c(i) = a(i) + b(i) + p
  end do
  !$omp end parallel do

  print *, c

end subroutine omp_lastprivate

subroutine omp_lastprivate_do(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !$omp parallel
  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp do lastprivate(p)
  do i = 1, 10
    c(i) = a(i) + b(i) + p
  end do
  !$omp end do
  !$omp end parallel

  print *, c

end subroutine omp_lastprivate_do

subroutine omp_lastprivate_simd(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel do simd lastprivate(p)
  do i = 1, 10
    c(i) = a(i) + b(i) + p
  end do
  !$omp end parallel do simd

  print *, c

end subroutine omp_lastprivate_simd

subroutine omp_lastprivate_sections(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp sections lastprivate(p)
  !$omp section
    c = a + b + p
  !$omp end sections

  print *, c

end subroutine omp_lastprivate_sections

subroutine omp_lastprivate_parallel_sections(p)
  integer :: a(10), b(10), c(10)
  integer, pointer, intent(in) :: p

  a = 10
  b = 20

  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel sections lastprivate(p)
  !$omp section
    c = a + b + p
  !$omp end parallel sections

  print *, c

end subroutine omp_lastprivate_parallel_sections
