! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

subroutine both_omitted(a)
  integer :: a(1:0)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(:))
  !$omp end parallel
end subroutine

subroutine lower_omitted(a)
  integer :: a(1:0)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(:0))
  !$omp end parallel
end subroutine

subroutine upper_omitted(a)
  integer :: a(1:0)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(1:))
  !$omp end parallel
end subroutine

subroutine shifted_empty(a)
  integer :: a(5:4)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(:))
  !$omp end parallel
end subroutine

subroutine shifted_lower_omitted(a)
  integer :: a(5:4)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(:4))
  !$omp end parallel
end subroutine

subroutine second_dimension(a)
  integer :: a(1:3,5:4)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(2,:))
  !$omp end parallel
end subroutine

subroutine explicit_empty(a)
  integer :: a(1:0)
  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel reduction(+:a(1:0))
  !$omp end parallel
end subroutine

subroutine nonempty(a)
  integer :: a(1:3)
  !$omp parallel reduction(+:a(:))
  !$omp end parallel
end subroutine

subroutine singleton(a)
  integer :: a(1:3)
  !$omp parallel reduction(+:a(2:2))
  !$omp end parallel
end subroutine

subroutine unknown_dynamic(a)
  integer :: a(:)
  !$omp parallel reduction(+:a(:))
  !$omp end parallel
end subroutine
