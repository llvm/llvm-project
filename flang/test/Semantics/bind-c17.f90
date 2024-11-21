! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
module m
  type a ! not BIND(C)
  end type
 contains
  subroutine sub(x) bind(c)
    !PORTABILITY: The derived type of this interoperable object should be BIND(C)
    type(a), pointer, intent(in) :: x
  end
end
