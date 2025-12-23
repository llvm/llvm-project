!RUN: %python %S/../test_errors.py %s %flang -Werror -fopenmp -famd-allow-threadprivate-equivalence

program equiv
    implicit none
    common/ba/a,b,c
    common/bb/e,d,f
    integer :: a,b,c
    integer :: e,d,f
    integer :: x,y,z

    !WARNING: Variable 'a' from common block 'ba' appears in an EQUIVALENCE statement and a THREADPRIVATE directive, which does not conform to the OpenMP API specification.
    !$omp threadprivate(/ba/)

    equivalence (x,a)

    !$omp parallel num_threads(2)
        x = -42
        !$omp masked
            x = 42
        !$omp end masked
        !$omp barrier
        !$omp atomic update
            a = a + 1
        !$omp end atomic
        !$omp barrier
        print *, a
    !$omp end parallel
end program equiv
