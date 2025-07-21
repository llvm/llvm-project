!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

program test
!ERROR: PARALLEL DO is not a cancellable construct
!$omp cancel parallel do
end
