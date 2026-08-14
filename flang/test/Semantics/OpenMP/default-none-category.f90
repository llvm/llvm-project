!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

program omp_default_none_category
  integer :: a
  integer, pointer :: b
  integer, allocatable :: c
  integer :: d(10)

  !$omp parallel default(none:scalar)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
    print *, a
    print *, b
    print *, c
    print *, d
  !$omp end parallel

  !$omp parallel default(none:pointer)
    print *, a
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
    print *, b
    print *, c
    print *, d

  !$omp end parallel

  !$omp parallel default(none:allocatable)
    print *, a
    print *, b
  !ERROR: The DEFAULT(NONE) clause requires that 'c' must be listed in a data-sharing attribute clause
    print *, c
    print *, d

  !$omp end parallel
 
  !$omp parallel default(none:aggregate)
    print *, a
    print *, b
    print *, c
  !ERROR: The DEFAULT(NONE) clause requires that 'd' must be listed in a data-sharing attribute clause
    print *, d
  !$omp end parallel

  !$omp parallel default(none:all)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-sharing attribute clause
    print *, a
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-sharing attribute clause
    print *, b
  !ERROR: The DEFAULT(NONE) clause requires that 'c' must be listed in a data-sharing attribute clause
    print *, c
  !ERROR: The DEFAULT(NONE) clause requires that 'd' must be listed in a data-sharing attribute clause
    print *, d
  !$omp end parallel
  end program omp_default_none_category
