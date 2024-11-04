! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp 
! Test that there are no errors for an empty critical construct

!$omp critical
!$omp end critical
end
