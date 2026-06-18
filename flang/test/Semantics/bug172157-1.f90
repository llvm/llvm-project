!RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type t
    !ERROR: Passed-object dummy argument 'this' of procedure 'pp1' used as procedure pointer component interface may not have the POINTER attribute
    procedure(sub), pass, pointer :: pp1 => sub
    !ERROR: Passed-object dummy argument 'that' of procedure 'pp2' may not have the POINTER attribute unless INTENT(IN)
    procedure(sub), pass(that), pointer :: pp2 => sub
   contains
    procedure :: goodtbp => sub
    !ERROR: Passed-object dummy argument 'that' of procedure 'badtbp' may not have the POINTER attribute unless INTENT(IN)
    procedure, pass(that) :: badtbp => sub
  end type
 contains
  subroutine sub(this, that)
    class(t), pointer, intent(in) :: this
    class(t), pointer :: that
  end
end

program test
  use m
  type(t) xnt
  type(t), target :: xt
  !ERROR: In assignment to object dummy argument 'this=', the target 'xnt' is not an object with POINTER or TARGET attributes
  call xnt%goodtbp(null())
  call xt%goodtbp(null()) ! ok
end
