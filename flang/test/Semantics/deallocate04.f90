! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Check for type errors in DEALLOCATE statements

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x

Real :: r
Integer :: s
Integer :: e
Integer :: pi
Character(256) :: ee
Procedure(Real) :: prp

Allocate(x)

!ERROR: Must have CHARACTER type, but is INTEGER(4)
Deallocate(x, stat=s, errmsg=e)

!ERROR: Must have INTEGER type, but is REAL(4)
!ERROR: Must have CHARACTER type, but is INTEGER(4)
Deallocate(x, stat=r, errmsg=e)

End Program
