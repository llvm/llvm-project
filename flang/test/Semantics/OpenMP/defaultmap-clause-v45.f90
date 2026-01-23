!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45 -Werror

subroutine f00
!WARNING: 'variable-category' modifier is required
  !$omp target defaultmap(tofrom)
  !$omp end target
end

subroutine f01
!WARNING: AGGREGATE is not allowed in OpenMP v4.5, try -fopenmp-version=50
  !$omp target defaultmap(tofrom:aggregate)
  !$omp end target
end

subroutine f02
!WARNING: FROM is not allowed in OpenMP v4.5, try -fopenmp-version=50
  !$omp target defaultmap(from:scalar)
  !$omp end target
end

subroutine f03
!WARNING: ALL is not allowed in OpenMP v4.5, try -fopenmp-version=52
  !$omp target defaultmap(tofrom:all)
  !$omp end target
end

subroutine f04
!WARNING: FROM is not allowed in OpenMP v4.5, try -fopenmp-version=50
!WARNING: POINTER is not allowed in OpenMP v4.5, try -fopenmp-version=50
  !$omp target defaultmap(from:pointer)
  !$omp end target
end


