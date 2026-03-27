! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
!
! OpenMP Version 5.2
!
! 2.10.2 single Construct
! Copyprivate and Nowait clauses are allowed in both clause and end clause

subroutine omp_single
    integer, save :: i
    integer       :: j
    i = 10; j = 11

    !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
    !$omp single copyprivate(i) nowait
        print *, "omp single", i
    !$omp end single

    !$omp parallel private(i)
        !$omp single copyprivate(i)
            print *, "omp single", i
        !$omp end single
    !$omp end parallel

    !$omp parallel
        !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
        !$omp single nowait
            print *, "omp single", i
        !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
        !$omp end single copyprivate(i)

        !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
        !$omp single copyprivate(i)
            print *, "omp single", i
        !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
        !$omp end single nowait

        !ERROR: COPYPRIVATE variable 'j' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
        !$omp single private(j) copyprivate(j)
            print *, "omp single", j
        !ERROR: COPYPRIVATE variable 'j' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
        !WARNING: The COPYPRIVATE clause with 'j' is already used on the SINGLE directive [-Wopen-mp-usage]
        !$omp end single copyprivate(j)

        !$omp single nowait
            print *, "omp single", j
        !ERROR: At most one NOWAIT clause can appear on the SINGLE directive
        !$omp end single nowait
    !$omp end parallel

    !$omp single nowait
        print *, "omp single", i
    !$omp end single
end subroutine omp_single
