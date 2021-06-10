! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.4 workshare Construct
! Checks for OpenMP Parallel constructs enclosed in Workshare constructs

module workshare_mod
  interface assignment(=)
    module procedure work_assign
  end interface

  contains
    subroutine work_assign(a,b)
      integer, intent(out) :: a
      logical, intent(in) :: b(:)
    end subroutine work_assign

    integer function my_func()
      my_func = 10
    end function my_func

end module workshare_mod

program omp_workshare
  use workshare_mod

  integer, parameter :: n = 10
  integer :: i, j, a(10), b(10)
  integer, pointer :: p
  integer, target :: t
  logical :: l(10)
  real :: aa(n,n), bb(n,n), cc(n,n), dd(n,n), ee(n,n), ff(n,n)

  !$omp workshare

  !$omp parallel
  p => t
  a = l
  !$omp single
  ee = ff
  !$omp end single
  !$omp end parallel

  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp parallel sections
  !$omp section
  aa = my_func()
  !$omp end parallel sections

  !$omp parallel do
  do i = 1, 10
    b(i) = my_func() + i
  end do
  !$omp end parallel do

  !$omp parallel
  where (dd .lt. 5) dd = aa * my_func()
  !$omp end parallel

  !$omp end workshare

end program omp_workshare
