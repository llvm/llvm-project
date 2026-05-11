! RUN: %flang -fopenmp -c %s
! RUN: %flang_skip_driver -fopenmp -c %s

program hello_openmp
  implicit none
  integer :: i

  !$omp parallel private(i)
    do i = 0, 64
      print *, "Hello ", i
      !$omp barrier
    end do
  !$omp end parallel
end program
