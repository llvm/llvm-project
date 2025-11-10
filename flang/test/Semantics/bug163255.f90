!RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
module m
  type t
  end type
  interface operator (==)
    module procedure equal
  end interface
 contains
  logical function equal(b1, b2)
    class(t), pointer, intent(in) :: b1, b2
    equal = associated(b1, b2)
  end
end module

program test
  use m
  type(t), target :: target
  class(t), pointer :: p => target
  !CHECK: IF (equal(p,null(p))) STOP
  if (p == null(p)) stop
end
