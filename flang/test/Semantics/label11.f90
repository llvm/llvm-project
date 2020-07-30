! RUN: not %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: BLOCK DATA subprogram name mismatch
! CHECK: should be
! CHECK: FUNCTION name mismatch
! CHECK: SUBROUTINE name mismatch
! CHECK: PROGRAM name mismatch
! CHECK: SUBMODULE name mismatch
! CHECK: INTERFACE generic-name (t7) mismatch
! CHECK: mismatched INTERFACE
! CHECK: derived type definition name mismatch
! CHECK: MODULE PROCEDURE name mismatch
! CHECK: MODULE name mismatch
! C739 If END TYPE is followed by a type-name, the type-name shall be the
! same as that in the corresponding derived-type-stmt.

block data t1
end block data t2

function t3
end function t4

subroutine t9
end subroutine t10

program t13
end program t14

submodule (mod) t15
end submodule t16

module t5
  interface t7
  end interface t8
  type t17
  end type t18
contains
  module procedure t11
  end procedure t12
end module mox
