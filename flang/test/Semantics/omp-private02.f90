! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! OpenMP Version 4.5
! 2.15.3.3 private Clause
! Variables that appear in namelist statements may not appear in a private clause.

module test
  integer :: a, b, c
  namelist /nlist1/ a, b
end module

program omp_private
  use test

  integer :: p(10) ,q(10)
  namelist /nlist2/ c, d

  a = 5
  b = 10
  c = 100

  !ERROR: Variable 'a' in NAMELIST cannot be in a PRIVATE clause
  !ERROR: Variable 'c' in NAMELIST cannot be in a PRIVATE clause
  !$omp parallel private(a, c)
  d = a + b
  !$omp end parallel

  call sb()

  contains
    subroutine sb()
      namelist /nlist3/ p, q

      !ERROR: Variable 'p' in NAMELIST cannot be in a PRIVATE clause
      !ERROR: Variable 'd' in NAMELIST cannot be in a PRIVATE clause
      !$omp parallel private(p, d)
      p = c * b
      q = p * d
      !$omp end parallel

      write(*, nlist1)
      write(*, nlist2)
      write(*, nlist3)

    end subroutine

end program omp_private
