!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50 -Werror

subroutine f00
  !$omp target defaultmap(tofrom)
  !$omp end target
end

subroutine f01
  !$omp target defaultmap(tofrom:aggregate)
  !$omp end target
end

subroutine f02
  !$omp target defaultmap(from:scalar)
  !$omp end target
end

subroutine f03
!WARNING: ALL is not allowed in OpenMP v5.0, try -fopenmp-version=52
  !$omp target defaultmap(tofrom:all)
  !$omp end target
end

