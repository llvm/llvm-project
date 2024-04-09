! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! This tests the various semantics related to the clauses of various OpenMP atomic constructs

program OmpAtomic
    use omp_lib
    integer :: g, x

    !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
    !$omp atomic relaxed, seq_cst
        x = x + 1
    !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
    !$omp atomic read seq_cst, relaxed
        x = g
    !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
    !$omp atomic write relaxed, release
        x = 2 * 4
    !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
    !$omp atomic update release, seq_cst
    !ERROR: Invalid or missing operator in atomic update statement
        x = 10
    !ERROR: More than one memory order clause not allowed on OpenMP Atomic construct
    !$omp atomic capture release, seq_cst
        x = g
        g = x * 10
    !$omp end atomic
end program OmpAtomic
