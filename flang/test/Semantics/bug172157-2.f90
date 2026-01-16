!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
module m
  type t
    integer :: n = 0
   contains
    procedure :: tbp => f
  end type
 contains
  function f(this)
    class(t), pointer, intent(in) :: this
    integer, pointer :: f
    f => this%n
  end
end

program test
  use m
  type(t), target :: xt
  type(t), pointer :: xp
  xt%n = 1
!CHECK: PRINT *, f(xt)
  print *, xt%tbp()
!CHECK: f(xt)=2_4
  xt%tbp() = 2
  print *, xt%n
  xp => xt
!CHECK: PRINT *, f(xp)
  print *, xp%tbp()
!CHECK: f(xp)=3_4
  xp%tbp() = 3
  print *, xp%n
  print *, xt%n
end
