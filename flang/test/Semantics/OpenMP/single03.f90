! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
!
subroutine omp_single
    integer, save :: i
    integer       :: j
    i = 10; j = 11

    !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
    !$omp single copyprivate(i) nowait
        print *, "omp single", i
    !$omp end single

    !$omp parallel private(i)
        !$omp single copyprivate(i)
            print *, "omp single", i
        !$omp end single
    !$omp end parallel

    !$omp parallel
        !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
        !$omp single nowait
            print *, "omp single", i
        !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
        !$omp end single copyprivate(i)

        !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
        !$omp single copyprivate(i)
            print *, "omp single", i
        !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
        !$omp end single nowait

        !ERROR: COPYPRIVATE variable 'j' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
        !$omp single private(j) copyprivate(j)
            print *, "omp single", j
        !ERROR: COPYPRIVATE variable 'j' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
        !ERROR: 'j' appears more than once in a COPYPRIVATE clause
        !$omp end single copyprivate(j)

        !$omp single nowait
            print *, "omp single", j
        !ERROR: At most one NOWAIT clause can appear on SINGLE directive
        !$omp end single nowait
    !$omp end parallel

    !$omp single nowait
        print *, "omp single", i
    !$omp end single
end subroutine omp_single
