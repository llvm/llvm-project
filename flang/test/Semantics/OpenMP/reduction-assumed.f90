! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! Types for built in reductions must have types which are valid for the
! initialization and combiner expressions in the spec. This implies assumed
! rank and assumed size cannot be used.

subroutine assumedRank1(a)
  integer :: a(..)

  ! ERROR: The type of 'a' is incompatible with the reduction operator.
  !$omp parallel reduction(+:a)
  !$omp end parallel
end

subroutine assumedRank2(a)
  integer :: a(..)

  ! ERROR: The type of 'a' is incompatible with the reduction operator.
  !$omp parallel reduction(min:a)
  !$omp end parallel
end

subroutine assumedRank3(a)
  integer :: a(..)

  ! ERROR: The type of 'a' is incompatible with the reduction operator.
  !$omp parallel reduction(iand:a)
  !$omp end parallel
end

subroutine assumedSize1(a)
  integer :: a(*)

  ! ERROR: Whole assumed-size array 'a' may not appear here without subscripts
  !$omp parallel reduction(+:a)
  !$omp end parallel
end

subroutine assumedSize2(a)
  integer :: a(*)

  ! ERROR: Whole assumed-size array 'a' may not appear here without subscripts
  !$omp parallel reduction(max:a)
  !$omp end parallel
end

subroutine assumedSize3(a)
  integer :: a(*)

  ! ERROR: Whole assumed-size array 'a' may not appear here without subscripts
  !$omp parallel reduction(ior:a)
  !$omp end parallel
end
