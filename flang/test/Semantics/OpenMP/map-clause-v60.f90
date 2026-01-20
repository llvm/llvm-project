!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(a)
  integer :: a(*)
  ! No diagnostic expected, assumed-size arrays are allowed on MAP in 6.0.
  !$omp target map(a)
  !$omp end target
end

subroutine f01
  integer :: x
  ! No diagnostic expected, alloc is allowed on map-entering constructs
  !$omp target map(alloc: x)
  !$omp end target
end

subroutine f02
  integer :: x
  ! No diagnostic expected, release is allowed on map-exiting constructs
  !$omp target_data map(release: x)
  !$omp end target_data
end

subroutine f03
  integer :: x
  ! No diagnostic expected, delete is its own modifier in 6.0+
  !$omp target_data map(delete: x)
  !$omp end target_data
end

subroutine f04
  integer :: x
  !ERROR: Only the FROM, RELEASE, STORAGE, TOFROM map types are permitted for MAP clauses on the TARGET_EXIT_DATA directive
  !$omp target_exit_data map(alloc: x)
end
