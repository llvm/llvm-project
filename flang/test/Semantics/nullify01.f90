! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test that NULLIFY works

Module share
  Real, Pointer :: rp
  Procedure(Real), Pointer :: mprp
End Module share

Program nullifytest
Use share

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt),Pointer :: p
End Type

Type(t),Allocatable :: x(:)
Type(t),Pointer :: y(:)
Type(t),Pointer :: z

Integer, Pointer :: pi
Procedure(Real), Pointer :: prp

Allocate(rp)
Nullify(rp)

Allocate(x(3))
Nullify(x(2)%p)

Nullify(y(2)%p)

Nullify(pi)
Nullify(prp)
Nullify(mprp)

Nullify(z%p)

End Program
