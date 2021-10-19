! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

type :: t
end type
integer :: acos
double precision :: cos
!CHECK: Explicit type declaration ignored for intrinsic function 'int'
complex :: int
character :: sin
logical :: asin
type(t) :: atan
!CHECK: INTRINSIC statement for explicitly-typed 'int'
intrinsic int
!CHECK: The result type 'REAL(4)' of the intrinsic function 'acos' is not the explicit declared type 'INTEGER(4)'
!CHECK: Ignored declaration of intrinsic function 'acos'
print *, acos(0.)
!CHECK: The result type 'REAL(4)' of the intrinsic function 'cos' is not the explicit declared type 'REAL(8)'
!CHECK: Ignored declaration of intrinsic function 'cos'
print *, cos(0.)
!CHECK: The result type 'REAL(4)' of the intrinsic function 'sin' is not the explicit declared type 'CHARACTER(KIND=1,LEN=1_8)'
!CHECK: Ignored declaration of intrinsic function 'sin'
print *, sin(0.)
!CHECK: The result type 'REAL(4)' of the intrinsic function 'asin' is not the explicit declared type 'LOGICAL(4)'
!CHECK: Ignored declaration of intrinsic function 'asin'
print *, asin(0.)
!CHECK: The result type 'REAL(4)' of the intrinsic function 'atan' is not the explicit declared type 't'
!CHECK: Ignored declaration of intrinsic function 'atan'
print *, atan(0.)
end
