! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
!
program single
    ! Valid testcases
    !$omp single
        print *, x
    !$omp end single

    !$omp single nowait
        print *, x
    !$omp end single

    !$omp single copyprivate(x, y, z)
        print *, x
    !$omp end single

    !$omp single
        print *, x
    !$omp end single copyprivate(x)

    ! Invalid testcases
    !$omp single
        print *, x
    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
    !$omp end single copyprivate(x) nowait

    !ERROR: 'x' appears more than once in a COPYPRIVATE clause
    !$omp single copyprivate(x) copyprivate(x)
        print *, x
    !$omp end single

    !$omp single
        print *, x
    !ERROR: 'x' appears more than once in a COPYPRIVATE clause
    !$omp end single copyprivate(x) copyprivate(x)

    !ERROR: At most one NOWAIT clause can appear on SINGLE directive
    !$omp single nowait nowait
        print *, x
    !$omp end single

    !$omp single
        print *, x
    !ERROR: At most one NOWAIT clause can appear on SINGLE directive
    !$omp end single nowait nowait

    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
    !$omp single copyprivate(x) nowait
        print *, x
    !ERROR: 'x' appears more than once in a COPYPRIVATE clause
    !ERROR: At most one NOWAIT clause can appear on SINGLE directive
    !$omp end single copyprivate(x) nowait

    !$omp single copyprivate(x)
        print *, x
    !ERROR: 'x' appears more than once in a COPYPRIVATE clause
    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
    !$omp end single copyprivate(x) nowait

    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
    !$omp single copyprivate(x, y) nowait
        print *, x
    !ERROR: 'x' appears more than once in a COPYPRIVATE clause
    !ERROR: 'z' appears more than once in a COPYPRIVATE clause
    !ERROR: At most one NOWAIT clause can appear on SINGLE directive
    !$omp end single copyprivate(x, z) copyprivate(z) nowait

    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on SINGLE directive
    !$omp single copyprivate(x) nowait copyprivate(y) copyprivate(z)
        print *, x
    !ERROR: 'x' appears more than once in a COPYPRIVATE clause
    !ERROR: 'y' appears more than once in a COPYPRIVATE clause
    !ERROR: 'z' appears more than once in a COPYPRIVATE clause
    !ERROR: At most one NOWAIT clause can appear on SINGLE directive
    !$omp end single copyprivate(x, y, z) nowait
end program
