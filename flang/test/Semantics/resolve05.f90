! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
program p
  integer :: p ! this is ok
end
module m
  integer :: m ! this is ok
end
submodule(m) sm
  integer :: sm ! this is ok
end
module m2
  type :: t
  end type
  interface
    subroutine s
      !ERROR: Module 'm2' cannot USE itself
      use m2, only: t
    end subroutine
  end interface
end module
subroutine s
  !ERROR: 's' is already declared in this scoping unit
  integer :: s
end
function f() result(res)
  integer :: res
  !ERROR: 'f' is already declared in this scoping unit
  !ERROR: The type of 'f' has already been declared
  real :: f
  res = 1
end
