! Check that the appropriate warnings are emitted when using INLINEALWAYS incorrectly
! RUN: %python %S/test_errors.py %s %flang_fc1

module m
! ERROR: !DIR$ INLINEALWAYS directive with name must appear in a subroutine or function
  !DIR$ INLINEALWAYS m
end module

subroutine test_subroutine()
! WARNING: INLINEALWAYS name does not match the subroutine or function name [-Wignored-directive]
  !DIR$ INLINEALWAYS wrong_subroutine
end subroutine

function test_func()
! WARNING: INLINEALWAYS name does not match the subroutine or function name [-Wignored-directive]
  !DIR$ INLINEALWAYS wrong_func
end function
