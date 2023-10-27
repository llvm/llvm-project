! RUN: %python %S/../test_errors.py %s %flang -fopenmp
!XFAIL: *
! OpenMP version 5.0.0
! 2.8.1 sections construct
! Orphaned section directives are prohibited. That is, the section directives must appear within the sections construct and must not be encountered elsewhere in the sections region
!TODO: Error in parsing. Make parser errors more informative. Until then, the test is XFAIL

program OmpOrphanedSections
   use omp_lib
   integer counter
   counter = 0
   !CHECK: expected 'END'
   !CHECK: END PROGRAM statement
   !CHECK: in the context: main program
   !CHECK: expected 'END PROGRAM'
   !CHECK: in the context: END PROGRAM statement
   !CHECK: in the context: main program
   !$omp section
   print *, "An orphaned section containing a single statement"
   !$omp section
   counter = counter + 1
   print *, "An orphaned section containing multiple statements"
!$omp sections
   !$omp section
   print *, "Not an orphan structured block"
!$omp end sections
end program OmpOrphanedSections
