! Check the errors in the directives of a plugin loaded with `flang -fc1 -load`
! for the loop that follows them.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: not %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -fsyntax-only %s 2>&1 | FileCheck %s

subroutine s(x)
  real :: x
  integer :: i
  ! CHECK: error: A 'example converge' directive needs at least 1 variable(s)
  !dir$ example converge(tol=1.0)
  do i = 1, 2
  end do
  ! CHECK: error: 's' is not a variable
  !dir$ example converge(s)
  do i = 1, 2
  end do
  ! CHECK: error: Argument 'max_iters' must be an integer
  !dir$ example converge(x) max_iters(1.5)
  do i = 1, 2
  end do
  ! CHECK: error: The variables of a 'example converge' directive must come before its other arguments
  !dir$ example converge(x, tol=1.0, x)
  do i = 1, 2
  end do
  !dir$ example converge(x)
  x = 1.
end
