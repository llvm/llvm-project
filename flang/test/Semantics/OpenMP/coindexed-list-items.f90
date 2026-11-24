! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

subroutine coindexed_noncontiguous
  integer, save :: a(3)[*]
  !ERROR: A coindexed object is not allowed as a list item on REDUCTION clause
  !$omp parallel reduction(+:a(1:3:2)[1])
  !$omp end parallel
end

subroutine noncoindexed_noncontiguous
  integer :: a(3)
  !ERROR: Reference to 'a' must be a contiguous object
  !$omp parallel reduction(+:a(1:3:2))
  !$omp end parallel
end

subroutine noncoindexed_contiguous
  integer :: a(3)
  !$omp parallel reduction(+:a(1:3))
  !$omp end parallel
end

subroutine list_items
  integer, save :: x[*], a(3)[*]
  type t
    integer :: n
  end type
  type(t), save :: obj[*]
  !ERROR: A coindexed object is not allowed as a list item on REDUCTION clause
  !$omp parallel reduction(+:x[1])
  !$omp end parallel
  !ERROR: A coindexed object is not allowed as a list item on REDUCTION clause
  !$omp parallel reduction(+:a(:)[1])
  !$omp end parallel
  !ERROR: A coindexed object is not allowed as a list item on SHARED clause
  !$omp parallel shared(x[1])
  !$omp end parallel
  !ERROR: A coindexed object is not allowed as a list item on PRIVATE clause
  !$omp parallel private(x[1])
  !$omp end parallel
  !ERROR: A coindexed object is not allowed as a list item on MAP clause
  !$omp target map(to:obj[1]%n)
  !$omp end target
  !ERROR: A coindexed object is not allowed as a list item on AFFINITY clause
  !$omp task affinity(x[1])
  !$omp end task
  !ERROR: A coindexed object is not allowed as a list item on FLUSH directive
  !$omp flush(x[1])
  !$omp parallel shared(x,a)
  !$omp end parallel
  !$omp target map(to:a(x[1]))
  !$omp end target
  !$omp parallel if(x[1] > 0)
  !$omp end parallel
end

subroutine task_reductions
  integer, save :: x[*]
  !ERROR: A coindexed object is not allowed as a list item on TASK_REDUCTION clause
  !$omp taskgroup task_reduction(+:x[1])
  !ERROR: A coindexed object is not allowed as a list item on IN_REDUCTION clause
  !$omp task in_reduction(+:x[1])
  !$omp end task
  !$omp end taskgroup
end

! Keep separate types so erroneous symbols cannot mask another clause's crash.
subroutine malformed_coindexed_affinity
  type t
    integer :: n
  end type
  type(t), save :: obj[*]
  !ERROR: 'n' is not an array
  !ERROR: A coindexed object is not allowed as a list item on AFFINITY clause
  !$omp task affinity(obj[1]%n(1:2))
  !$omp end task
end

! Keep separate types so erroneous symbols cannot mask another clause's crash.
subroutine malformed_coindexed_depend
  type t
    integer :: n
  end type
  type(t), save :: obj[*]
  !ERROR: 'n' is not an array
  !ERROR: A coindexed object is not allowed as a list item on DEPEND clause
  !$omp task depend(in:obj[1]%n(1:2))
  !$omp end task
end
