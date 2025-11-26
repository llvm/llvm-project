! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! Test mapper usage in target update to/from clauses

program target_update_mapper
  implicit none

  integer, parameter :: n = 4

  type :: typ
    integer, allocatable :: a(:)
    integer, allocatable :: b(:)
  end type typ

  !$omp declare mapper(custom: typ :: t) map(t%a)

  type(typ) :: t
  integer :: not_a_mapper
  allocate(t%a(n), source=1)
  allocate(t%b(n), source=2)

  !$omp target enter data map(alloc: t)

  ! Valid: using custom mapper with target update to
  t%a = 42
  !$omp target update to(mapper(custom): t)

  !$omp target
    t%a(:) = t%a(:) / 2
    t%b(:) = -1
  !$omp end target

  ! Valid: using custom mapper with target update from
  !$omp target update from(mapper(custom): t)

  ! Valid: using default mapper explicitly
  !$omp target update to(mapper(default): t)

  print*, t%a
  print*, t%b

  !$omp target exit data map(delete: t)
  deallocate(t%a)
  deallocate(t%b)

  ! Test error case: undefined mapper
  !ERROR: 'undefined_mapper' not declared
  !$omp target update to(mapper(undefined_mapper): t)

  ! Test error case: wrong kind of symbol
  !ERROR: Name 'not_a_mapper' should be a mapper name
  !$omp target update from(mapper(not_a_mapper): t)

end program target_update_mapper
