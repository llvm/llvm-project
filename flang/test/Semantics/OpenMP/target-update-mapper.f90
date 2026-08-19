! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! Test mapper name resolution in target update to/from clauses

program target_update_mapper
  type :: typ
    integer :: a
  end type typ

  !$omp declare mapper(custom: typ :: t) map(t%a)

  type(typ) :: t
  integer :: not_a_mapper

  ! Valid: using custom mapper
  !$omp target update to(mapper(custom): t)
  !$omp target update from(mapper(custom): t)

  ! Valid: using default mapper
  !$omp target update to(mapper(default): t)

  ! Error: undefined mapper
  !ERROR: 'undefined_mapper' not declared
  !$omp target update to(mapper(undefined_mapper): t)

  ! Error: wrong kind of symbol
  !ERROR: Name 'not_a_mapper' should be a mapper name
  !$omp target update from(mapper(not_a_mapper): t)

end program target_update_mapper
