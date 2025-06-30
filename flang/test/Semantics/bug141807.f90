!RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
!Ensure that non-elemental specific takes precedence over elemental
!defined assignment, even with non-default PASS argument.
module m
  type base
    integer :: n = -999
   contains
    procedure, pass(from) :: array_assign_scalar
    procedure :: elemental_assign
    generic :: assignment(=) => array_assign_scalar, elemental_assign
  end type
 contains
  subroutine array_assign_scalar(to, from)
    class(base), intent(out) :: to(:)
    class(base), intent(in) :: from
    to%n = from%n
  end
  impure elemental subroutine elemental_assign(to, from)
    class(base), intent(out) :: to
    class(base), intent(in) :: from
    to%n = from%n
  end
end

use m
type(base) :: array(1), scalar
scalar%n = 1
!CHECK: CALL array_assign_scalar(array,(scalar))
array = scalar
!CHECK: CALL elemental_assign(array,[base::scalar])
array = [scalar]
end
