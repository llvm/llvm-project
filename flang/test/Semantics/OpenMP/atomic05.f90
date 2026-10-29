! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=50

! This tests the various semantics related to the clauses of various OpenMP atomic constructs

program OmpAtomic
    integer :: g, x

    !ERROR: RELAXED and SEQ_CST clauses are mutually exclusive as members of 'memory-order' clause group
    !$omp atomic relaxed, seq_cst
        x = x + 1
    !ERROR: SEQ_CST and RELAXED clauses are mutually exclusive as members of 'memory-order' clause group
    !$omp atomic read seq_cst, relaxed
        x = g
    !ERROR: RELAXED and RELEASE clauses are mutually exclusive as members of 'memory-order' clause group
    !$omp atomic write relaxed, release
        x = 2 * 4
    !ERROR: RELEASE and SEQ_CST clauses are mutually exclusive as members of 'memory-order' clause group
    !$omp atomic update release, seq_cst
    !ERROR: This is not a valid ATOMIC UPDATE operation
        x = 10
    !ERROR: RELEASE and SEQ_CST clauses are mutually exclusive as members of 'memory-order' clause group
    !$omp atomic capture release, seq_cst
        x = g
        g = x * 10
    !$omp end atomic
end program OmpAtomic
