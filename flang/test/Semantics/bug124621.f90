! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type t1
   contains
    procedure, pass(from) :: defAsst1
    generic :: assignment(=) => defAsst1
  end type
  type t2
  end type
  type t3
  end type
  interface assignment(=)
    module procedure defAsst2
  end interface
 contains
  subroutine defAsst1(to,from)
    class(*), intent(out) :: to
    class(t1), intent(in) :: from
  end
  subroutine defAsst2(to,from)
    class(*), intent(out) :: to
    class(t2), intent(in) :: from
  end
end

program test
  use m
  type(t1) x1
  type(t2) x2
  type(t3) x3
  j = x1
  j = x2
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types INTEGER(4) and TYPE(t3)
  j = x3
  x1 = x1
  x1 = x2
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(t1) and TYPE(t3)
  x1 = x3
  x2 = x1
  x2 = x2
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(t2) and TYPE(t3)
  x2 = x3
  x3 = x1
  x3 = x2
  x3 = x3
end
