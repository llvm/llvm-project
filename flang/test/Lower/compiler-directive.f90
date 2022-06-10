! RUN: bbc %s -o - 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - 2>&1 | FileCheck %s

! CHECK: ignoring all compiler directives

MODULE test_mod
  CONTAINS
  SUBROUTINE foo()
    REAL :: var
  !DIR$ VECTOR ALIGNED
    var = 1.
  END SUBROUTINE foo
END MODULE test_mod
