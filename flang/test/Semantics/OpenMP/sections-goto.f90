! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Regression test for #143231

!$omp sections
! ERROR: invalid branch into an OpenMP structured block
! ERROR: invalid branch leaving an OpenMP structured block
goto 10
!$omp section
10 print *, "Invalid jump"
!$omp end sections
end
