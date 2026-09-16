! Check that the appropriate warnings are emitted when using INLINEALWAYS incorrectly
! RUN: %python %S/test_errors.py %s %flang_fc1

module m
! ERROR: !DIR$ INLINEALWAYS directive with name must appear in a subprogram
  !DIR$ INLINEALWAYS m
end module

subroutine test_subroutine()
! WARNING: INLINEALWAYS name 'wrong_subroutine' does not match the subprogram name 'test_subroutine' [-Wignored-directive]
  !DIR$ INLINEALWAYS wrong_subroutine
end subroutine

function test_func()
! WARNING: INLINEALWAYS name 'wrong_func' does not match the subprogram name 'test_func' [-Wignored-directive]
  !DIR$ INLINEALWAYS wrong_func
end function
