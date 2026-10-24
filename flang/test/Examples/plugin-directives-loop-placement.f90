! Check that a directive of a plugin loaded with `flang -fc1 -load` for the
! loop that follows it must come in front of a DO or DO WHILE loop. (Where
! the directive is is checked after names are resolved, and only if they
! are, hence a test of its own.)

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: not %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -fsyntax-only %s 2>&1 | FileCheck %s

subroutine s(x)
  real :: x
  integer :: i
  ! CHECK: error: A DO or DO WHILE loop must follow the 'EXAMPLE CONVERGE' directive
  !dir$ example converge(x)
  do concurrent (i = 1:2)
  end do
  ! CHECK: error: A DO or DO WHILE loop must follow the 'EXAMPLE CONVERGE' directive
  !dir$ example converge(x)
  x = 1.
end
