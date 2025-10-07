! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=51

! OpenMP Version 5.0: 2.10.1
! Various checks for DETACH Clause

program detach02
    use omp_lib, only: omp_event_handle_kind
    integer(omp_event_handle_kind)          :: event_01, event_02

    !ERROR: At most one DETACH clause can appear on the TASK directive
    !$omp task detach(event_01) detach(event_02)
       x = x + 1
    !$omp end task

    !ERROR: Clause MERGEABLE is not allowed if clause DETACH appears on the TASK directive
    !$omp task detach(event_01) mergeable
        x = x + 1
    !$omp end task
end program
