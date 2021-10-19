! RUN: %python %S/test_errors.py %s %flang_fc1
! Test that DEALLOCATE works

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
Integer :: s
CHARACTER(256) :: e

Integer, Pointer :: pi

Allocate(pi)
Allocate(x(3))

Deallocate(x(2)%p)

Deallocate(y(2)%p)

Deallocate(pi)

Deallocate(z%p)

Deallocate(x%p, stat=s, errmsg=e)
Deallocate(x%p, errmsg=e)
Deallocate(x%p, stat=s)

Deallocate(y%p, stat=s, errmsg=e)
Deallocate(y%p, errmsg=e)
Deallocate(y%p, stat=s)

Deallocate(z, stat=s, errmsg=e)
Deallocate(z, errmsg=e)
Deallocate(z, stat=s)

Deallocate(z, y%p, stat=s, errmsg=e)
Deallocate(z, y%p, errmsg=e)
Deallocate(z, y%p, stat=s)

End Program
