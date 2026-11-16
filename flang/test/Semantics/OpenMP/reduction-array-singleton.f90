! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine singleton(a)
  integer :: a(4)
  !$omp parallel reduction(+:a(2:2))
  !$omp end parallel
end subroutine

subroutine ordinary(a)
  integer :: a(4)
  !$omp parallel reduction(+:a(2:3))
  !$omp end parallel
end subroutine

subroutine rank_two(a)
  integer :: a(2,3)
  !$omp parallel reduction(+:a(1:2,2:2))
  !$omp end parallel
end subroutine

subroutine large_bound(a)
  integer :: a(9223372036854775805_8:9223372036854775807_8)
  !$omp parallel reduction(+:a(9223372036854775806_8:9223372036854775806_8))
  !$omp end parallel
end subroutine

subroutine unknown_bound(a, l, u)
  integer :: a(:), l, u
  !$omp parallel reduction(+:a(l:u))
  !$omp end parallel
end subroutine

subroutine empty(a)
  integer :: a(4)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(2:1))
  !$omp end parallel
end subroutine
