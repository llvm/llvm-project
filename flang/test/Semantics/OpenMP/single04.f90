! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
!
! OpenMP Version 5.2
!
! 2.10.2 single Construct
! Valid and invalid testcases for copyprivate and nowait clause on the single directive

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
    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
    !$omp end single copyprivate(x) nowait

    !ERROR: 'x' appears in more than one COPYPRIVATE clause on the SINGLE directive
    !$omp single copyprivate(x) copyprivate(x)
        print *, x
    !$omp end single

    !$omp single
        print *, x
    !ERROR: 'x' appears in more than one COPYPRIVATE clause on the END SINGLE directive
    !$omp end single copyprivate(x) copyprivate(x)

    !ERROR: At most one NOWAIT clause can appear on the SINGLE directive
    !$omp single nowait nowait
        print *, x
    !$omp end single

    !$omp single
        print *, x
    !ERROR: At most one NOWAIT clause can appear on the END SINGLE directive
    !$omp end single nowait nowait

    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
    !$omp single copyprivate(x) nowait
        print *, x
    !WARNING: The COPYPRIVATE clause with 'x' is already used on the SINGLE directive [-Wopen-mp-usage]
    !ERROR: At most one NOWAIT clause can appear on the SINGLE directive
    !$omp end single copyprivate(x) nowait

    !$omp single copyprivate(x)
        print *, x
    !WARNING: The COPYPRIVATE clause with 'x' is already used on the SINGLE directive [-Wopen-mp-usage]
    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
    !$omp end single copyprivate(x) nowait

    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
    !$omp single copyprivate(x, y) nowait
        print *, x
    !WARNING: The COPYPRIVATE clause with 'x' is already used on the SINGLE directive [-Wopen-mp-usage]
    !ERROR: 'z' appears in more than one COPYPRIVATE clause on the END SINGLE directive
    !ERROR: At most one NOWAIT clause can appear on the SINGLE directive
    !$omp end single copyprivate(x, z) copyprivate(z) nowait

    !ERROR: NOWAIT clause must not be used with COPYPRIVATE clause on the SINGLE directive
    !$omp single copyprivate(x) nowait copyprivate(y) copyprivate(z)
        print *, x
    !WARNING: The COPYPRIVATE clause with 'x' is already used on the SINGLE directive [-Wopen-mp-usage]
    !WARNING: The COPYPRIVATE clause with 'y' is already used on the SINGLE directive [-Wopen-mp-usage]
    !WARNING: The COPYPRIVATE clause with 'z' is already used on the SINGLE directive [-Wopen-mp-usage]
    !ERROR: At most one NOWAIT clause can appear on the SINGLE directive
    !$omp end single copyprivate(x, y, z) nowait
end program
