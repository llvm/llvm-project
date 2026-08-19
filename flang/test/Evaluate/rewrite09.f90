!RUN: %flang_fc1 -fdebug-unparse -funsigned %s | FileCheck %s --check-prefix=UNPARSE

subroutine f00(x)
  integer :: x, y, z
  y = x + 0
  z = 0 + x
end

!UNPARSE: SUBROUTINE f00 (x)
!UNPARSE:  INTEGER x, y, z
!UNPARSE:   y=(x)
!UNPARSE:   z=(x)
!UNPARSE: END SUBROUTINE

subroutine f01(x)
  integer :: x, y, z
  y = x - 0
  z = 0 - x
end

!UNPARSE: SUBROUTINE f01 (x)
!UNPARSE:  INTEGER x, y, z
!UNPARSE:   y=(x)
!UNPARSE:   z=0_4-x
!UNPARSE: END SUBROUTINE

subroutine f02(x)
  unsigned :: x, y, z
  y = x + 0u
  z = 0u + x
end

!UNPARSE: SUBROUTINE f02 (x)
!UNPARSE:  UNSIGNED x, y, z
!UNPARSE:   y=(x)
!UNPARSE:   z=(x)
!UNPARSE: END SUBROUTINE

subroutine f03(x)
  unsigned :: x, y, z
  y = x - 0u
  z = 0u - x
end

!UNPARSE: SUBROUTINE f03 (x)
!UNPARSE:  UNSIGNED x, y, z
!UNPARSE:   y=(x)
!UNPARSE:   z=0U_4-x
!UNPARSE: END SUBROUTINE
