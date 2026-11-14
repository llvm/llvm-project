! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=45

subroutine bad_in_45(h_ptr)
  integer, pointer :: h_ptr
  !ERROR: USE_DEVICE_ADDR clause is not allowed on TARGET DATA directive in OpenMP v4.5, try -fopenmp-version=50 [-Wopenmp-future]
  !ERROR: MAP clause is required on TARGET DATA directive
  !$omp target data use_device_addr(h_ptr)
  !$omp end target data
end

subroutine f02(N)
  integer :: N
  integer :: i, j
  real :: a
  !ERROR: ORDERED clause with an argument is not allowed on a compound directive with SIMD as a constituent
  !$omp do simd ordered(2)
  do i = 1, N
     do j = 1, N
        a = 3.14
     enddo
  enddo
end
