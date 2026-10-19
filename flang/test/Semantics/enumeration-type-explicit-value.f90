! RUN: not %flang_fc1 -fsyntax-only -fenumeration-type %s 2>&1 | FileCheck %s
! F2023 enumeration types do not permit explicit enumerator values (unlike the
! ENUM, BIND(C) construct).  'enumerator :: red = 1' must be rejected.

subroutine test_explicit_value()
  ! CHECK: error: expected end of statement
  enumeration type :: color
    enumerator :: red = 1, green, blue
  end enumeration type
end subroutine
