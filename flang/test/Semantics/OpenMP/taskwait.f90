! UNSUPPORTED: system-windows
! Marking as unsupported due to suspected long runtime on Windows
! RUN: %python %S/../test_errors.py %s %flang -fopenmp

!$omp taskwait
end
