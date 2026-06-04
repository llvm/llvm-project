! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! OpenMP Version 5.2
! Check OpenMP construct validity for the following directives:
! 2.14.7 Declare Target Directive
! When used in an implicit none context.
! Per OpenMP 5.2 ,3.2.1, 7.8 & 7.8.1, names in DECLARE TARGET may denote
! procedures. Unknown names are treated as external procedures, so no
! "No explicit type" error is expected for names in ENTER/TO clauses
! (or bare list). LINK clause is different.

module test_0
    implicit none
!$omp declare target(no_implicit_materialization_1)

!ERROR: No explicit type declared for 'no_implicit_materialization_2'
!$omp declare target link(no_implicit_materialization_2)

!WARNING: The usage of TO clause on DECLARE TARGET directive has been deprecated. Use ENTER clause instead. [-Wopenmp-usage]
!$omp declare target to(no_implicit_materialization_3)

!ERROR: 'no_implicit_materialization_3' must be a variable or a procedure
!$omp declare target enter(no_implicit_materialization_3)

INTEGER :: data_int = 10
!$omp declare target(data_int)
end module test_0
