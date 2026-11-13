! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52 -Werror -Wno-experimental-option

subroutine f00(x)
  integer :: x
!ERROR: If a 'memory-order' clause is specified, list items must not be specified on the FLUSH directive
!ERROR: The syntax "FLUSH clause (object, ...)" has been deprecated, use "FLUSH(object, ...) clause" instead
  !$omp flush seq_cst (x)
end
