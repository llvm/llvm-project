! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=52 -Werror

subroutine f00(x)
  integer :: x
!ERROR: The syntax "FLUSH clause (object, ...)" has been deprecated, use "FLUSH(object, ...) clause" instead
  !$omp flush seq_cst (x)
end
