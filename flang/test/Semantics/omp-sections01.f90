! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.2 sections Construct
! Only a single nowait clause can appear on a sections directive.

program omp_sections

  !$omp sections
    !$omp section
    print *, "omp section"
  !ERROR: Only a single nowait clause can appear on a sections directive.
  !$omp end sections nowait nowait

end program omp_sections
