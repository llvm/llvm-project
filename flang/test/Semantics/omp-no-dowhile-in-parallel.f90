! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell

subroutine bug48308(x,i)
  real :: x(:)
  integer :: i
  !$omp parallel firstprivate(i)
    do while (i>0)
      x(i) = i
      i = i - 1
    end do
  !$omp end parallel
end subroutine

subroutine s1(x,i)
  real :: x(:)
  integer :: i
  !$omp parallel firstprivate(i)
    do i = 10, 1, -1
      x(i) = i
    end do
  !$omp end parallel

  !$omp parallel firstprivate(i)
    do concurrent (i = 1:10:1)
      x(i) = i
    end do
  !$omp end parallel
end subroutine
