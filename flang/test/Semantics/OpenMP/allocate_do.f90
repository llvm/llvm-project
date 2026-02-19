! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
module all_mod
    integer, parameter :: N = 100
    integer(kind=4), dimension(N) :: AAA
    integer(kind=4), dimension(N) :: BBB
    integer(kind=4), dimension(N) :: CCC
    integer(kind=4), dimension(N) :: DDD
    integer val, error_count
    contains
    subroutine initialize()
        do i=1,N
            AAA(i) = i
            BBB(i) = 2*i
            CCC(i) = 0
            DDD(i) = 0
        end do
        val = 0
    end subroutine
end module

subroutine test_omp_do()
    use all_mod
    !$omp parallel shared(AAA,BBB,CCC,DDD,val)
    !$omp do private(CCC, val) allocate(0:CCC, val)
    do i=1,N
        CCC(i) = AAA(i) + BBB(i)
        val    = AAA(i) + BBB(i)
        DDD(i) = CCC(i) + val
    end do
    !$omp end do
    !$omp end parallel
end subroutine test_omp_do

subroutine test_omp_parallel_do()
    use all_mod
    !$omp parallel do private(CCC, val) allocate(0:CCC, val) shared(AAA,BBB,DDD)
    do i=1,N
        CCC(i) = AAA(i) + BBB(i)
        val    = AAA(i) + BBB(i)
        DDD(i) = CCC(i) + val
    end do
    !$omp end parallel do
end subroutine test_omp_parallel_do
program test_omp_do
end program
