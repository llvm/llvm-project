! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.2
! 5.1.1 - Variables Referenced in a Construct
! Copyprivate must accept variables that are predetermined as private.

module m1
  integer :: m
end module

program omp_copyprivate
  use m1
  implicit none
  integer :: i
  integer, save :: j
  integer :: k
  common /c/ k
  real, parameter :: pi = 3.14
  integer :: a1(10)

  ! Local variables are private.
  !$omp single
    i = 123
  !$omp end single copyprivate(i)
  !$omp single
  !$omp end single copyprivate(a1)

  ! Variables with the SAVE attribute are not private.
  !$omp single
  !ERROR: COPYPRIVATE variable 'j' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(j)

  ! Common block variables are not private.
  !$omp single
  !ERROR: COPYPRIVATE variable 'k' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(/c/)
  !$omp single
  !ERROR: COPYPRIVATE variable 'k' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(k)

  ! Module variables are not private.
  !$omp single
  !ERROR: COPYPRIVATE variable 'm' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(m)

  ! Parallel can make a variable shared.
  !$omp parallel
    !$omp single
      i = 456
    !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
    !$omp end single copyprivate(i)
    call sub(j, a1)
  !$omp end parallel

  !$omp parallel shared(i)
    !$omp single
      i = 456
    !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
    !$omp end single copyprivate(i)
  !$omp end parallel

  !FIXME: an error should be emitted in this case.
  !       copyprivate(i) should be considered as a reference to i and a new
  !       symbol should be created in `parallel` scope, for this case to be
  !       handled properly.
  !$omp parallel
    !$omp single
    !$omp end single copyprivate(i)
  !$omp end parallel

  ! Named constants are shared.
  !$omp single
  !ERROR: COPYPRIVATE variable 'pi' is not PRIVATE or THREADPRIVATE in outer context
  !$omp end single copyprivate(pi)

  !$omp parallel do
  do i = 1, 10
    !$omp parallel
    !$omp single
      j = i
    !ERROR: COPYPRIVATE variable 'i' is not PRIVATE or THREADPRIVATE in outer context
    !$omp end single copyprivate(i)
    !$omp end parallel
  end do
  !$omp end parallel do

contains
  subroutine sub(s1, a)
    integer :: s1
    integer :: a(:)

    ! Dummy argument.
    !$omp single
    !$omp end single copyprivate(s1)

    ! Assumed shape arrays are shared.
    !$omp single
    !ERROR: COPYPRIVATE variable 'a' is not PRIVATE or THREADPRIVATE in outer context
    !$omp end single copyprivate(a)
  end subroutine

  integer function fun(f1)
    integer :: f1

    ! Dummy argument.
    !$omp single
    !$omp end single copyprivate(f1)

    ! Function result is private.
    !$omp single
    !$omp end single copyprivate(fun)
  end function
end program
