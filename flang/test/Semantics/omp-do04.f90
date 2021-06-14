! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a threadprivate directive.


program omp_do
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
end program omp_do

program omp_do1
  !$omp  threadprivate(k,j,i)
  !$omp  do
  !ERROR: Loop iteration variable i is not allowed in THREADPRIVATE.
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end program omp_do1

program omp_do2
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

end program omp_do2

program omp_do3
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

end program omp_do3

module tp
  !integer i,j
  integer, save:: i, j, k,n
  !$omp threadprivate(i)
  !$omp threadprivate(j)
end module tp

module usetp
  use tp
end module usetp

program main
  use usetp
  !$omp  do
  !ERROR: Loop iteration variable i is not allowed in THREADPRIVATE.
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do
end program

program main1
  use tp
  !$omp  do
  !ERROR: Loop iteration variable j is not allowed in THREADPRIVATE.
  do j = 1, 10
    do i = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do
end program
