! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=50

! This tests the various semantics related to the clauses of various OpenMP atomic constructs

program OmpAtomic
    use omp_lib
    integer :: g, x

    !ERROR: At most one clause from the 'memory-order' group is allowed on ATOMIC construct
    !$omp atomic relaxed, seq_cst
        x = x + 1
    !ERROR: At most one clause from the 'memory-order' group is allowed on ATOMIC construct
    !$omp atomic read seq_cst, relaxed
        x = g
    !ERROR: At most one clause from the 'memory-order' group is allowed on ATOMIC construct
    !$omp atomic write relaxed, release
        x = 2 * 4
    !ERROR: At most one clause from the 'memory-order' group is allowed on ATOMIC construct
    !$omp atomic update release, seq_cst
    !ERROR: The atomic variable x should appear as an argument in the update operation
        x = 10
    !ERROR: At most one clause from the 'memory-order' group is allowed on ATOMIC construct
    !$omp atomic capture release, seq_cst
        x = g
        g = x * 10
    !$omp end atomic
end program OmpAtomic
