! RUN: %S/test_errors.sh %s %t %f18
! C739 If END TYPE is followed by a type-name, the type-name shall be the
! same as that in the corresponding derived-type-stmt.
! C1401 The program-name shall not be included in the end-program-stmt unless
! the optional program-stmt is used. If included, it shall be identical to the
! program-name specified in the program-stmt.
! C1402 If the module-name is specified in the end-module-stmt, it shall be
! identical to the module-name specified in the module-stmt.
! C1413 If a submodule-name appears in the end-submodule-stmt, it shall be
! identical to the one in the submodule-stmt.
! C1414 If a function-name appears in the end-function-stmt, it shall be
! identical to the function-name specified in the function-stmt.
! C1502 If the end-interface-stmt includes a generic-spec, the interface-stmt
! shall specify the same generic-spec
! C1564 If a function-name appears in the end-function-stmt, it shall be
! identical to the function-name specified in the function-stmt.
! C1567 If a submodule-name appears in the end-submodule-stmt, it shall be
! identical to the one in the submodule-stmt.
! C1569 If the module-name is specified in the end-module-stmt, it shall be
! identical to the module-name specified in the module-stmt

block data t1
!ERROR: BLOCK DATA subprogram name mismatch
end block data t2

function t3
!ERROR: FUNCTION name mismatch
end function t4

subroutine t9
!ERROR: SUBROUTINE name mismatch
end subroutine t10

program t13
!ERROR: END PROGRAM name mismatch
end program t14

submodule (mod) t15
!ERROR: SUBMODULE name mismatch
end submodule t16

module t5
  interface t7
  !ERROR: INTERFACE generic-name (t7) mismatch
  end interface t8
  type t17
  !ERROR: derived type definition name mismatch
  end type t18

  abstract interface
    subroutine subrFront()
    !ERROR: SUBROUTINE name mismatch
    end subroutine subrBack
    function funcFront(x)
      real, intent(in) :: x
      real funcFront
    !ERROR: FUNCTION name mismatch
    end function funcBack
  end interface

contains
  module procedure t11
  !ERROR: MODULE PROCEDURE name mismatch
  end procedure t12
!ERROR: MODULE name mismatch
end module mox
