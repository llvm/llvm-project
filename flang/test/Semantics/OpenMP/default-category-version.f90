!RUN: %python %S/../test_errors.py %s %flang_fc1 -Werror -Wno-experimental-option -fopenmp -fopenmp-version=51

subroutine default_category_version
  integer :: s

!ERROR: 'variable-category' modifier is not supported on DEFAULT clause in OpenMP v5.1, try -fopenmp-version=60
  !$omp parallel default(private:scalar)
  s = 1
  !$omp end parallel
end subroutine
