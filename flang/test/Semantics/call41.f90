! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
module m
 contains
  subroutine unlimited(x)
    class(*), intent(in) :: x
  end
  subroutine test
    !PORTABILITY: passing Hollerith to unlimited polymorphic as if it were CHARACTER
    call unlimited(6HHERMAN)
    call unlimited('abc') ! ok
  end
end
