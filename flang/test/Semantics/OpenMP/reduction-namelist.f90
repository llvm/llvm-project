! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! This is not actually disallowed by the OpenMP standard, but it is not allowed
! for privatisation - this seems like an oversight.

module test
  integer :: a, b, c
  namelist /nlist1/ a, b
end module

program omp_reduction
  use test

  integer :: p(10) ,q(10)
  namelist /nlist2/ c, d

  a = 5
  b = 10
  c = 100

  !ERROR: Variable 'd' in NAMELIST cannot be in a REDUCTION clause
  !ERROR: Variable 'a' in NAMELIST cannot be in a REDUCTION clause
  !$omp parallel reduction(+:d) reduction(+:a)
  d = a + b
  a = d
  !$omp end parallel

  call sb()

  contains
    subroutine sb()
      namelist /nlist3/ p, q

      !ERROR: Variable 'p' in NAMELIST cannot be in a REDUCTION clause
      !ERROR: Variable 'q' in NAMELIST cannot be in a REDUCTION clause
      !$omp parallel reduction(+:p) reduction(+:q)
      p = c * b
      q = p * d
      !$omp end parallel

      write(*, nlist1)
      write(*, nlist2)
      write(*, nlist3)

    end subroutine

end program omp_reduction
