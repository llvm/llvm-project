!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

program omp_default_category_02
!ERROR: 'variable-category' modifier cannot occur multiple times
  !$omp parallel default(private:scalar, aggregate)
  !$omp end parallel
end program omp_default_category_02
