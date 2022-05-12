! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a threadprivate directive.


subroutine omp_do
  integer, save:: i, j, k,n
  !$omp  threadprivate(k,j,i)
  !$omp  do collapse(2)
  !ERROR: Loop iteration variable i is not allowed in THREADPRIVATE.
  do i = 1, 10
    !ERROR: Loop iteration variable j is not allowed in THREADPRIVATE.
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do
end subroutine omp_do

subroutine omp_do1
  integer, save :: i, j, k
  !$omp  threadprivate(k,j,i)
  !$omp  do
  !ERROR: Loop iteration variable i is not allowed in THREADPRIVATE.
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end subroutine omp_do1

subroutine omp_do2
  integer, save :: k, j
  !$omp threadprivate(k)
  !$omp threadprivate(j)
  call compute()
  contains
  subroutine compute()
  !$omp  do ordered(1) collapse(1)
  !ERROR: Loop iteration variable k is not allowed in THREADPRIVATE.
  foo: do k = 1, 10
    do i = 1, 10
      print *, "Hello"
    end do
  end do foo
  !$omp end do
  end subroutine

end subroutine omp_do2

subroutine omp_do3
  integer, save :: i
  !$omp  threadprivate(i)
  !$omp parallel
  print *, "parallel"
  !$omp end parallel
  !$omp  do
  !ERROR: Loop iteration variable i is not allowed in THREADPRIVATE.
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end subroutine omp_do3

module tp
  !integer i,j
  integer, save:: i, j, k,n
  !$omp threadprivate(i)
  !$omp threadprivate(j)
end module tp

module usetp
  use tp
end module usetp

subroutine main
  use usetp
  !$omp  do
  !ERROR: Loop iteration variable i is not allowed in THREADPRIVATE.
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do
end subroutine

subroutine main1
  use tp
  !$omp  do
  !ERROR: Loop iteration variable j is not allowed in THREADPRIVATE.
  do j = 1, 10
    do i = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do
end subroutine
