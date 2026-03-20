!RUN: %python %S/../test_errors.py %s %flang -Werror -fopenmp -famd-allow-threadprivate-equivalence

program equiv
    implicit none
    common/ba/a,b,c
    common/bb/e,d,f
    integer :: a,b,c
    integer :: e,d,f
    integer :: x,y,z

    !WARNING: A variable in a THREADPRIVATE directive used in an EQUIVALENCE statement is an OpenMP extension (variable 'a' from common block '/ba/') [-Wopenmp-threadprivate-equivalence]
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
