! RUN: %S/test_errors.sh %s %t %f18
! Check for semantic errors in NULLIFY statements

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x(:)

Integer :: pi
Procedure(Real) :: prp

Allocate(x(3))
!ERROR: component in NULLIFY statement must have the POINTER attribute
Nullify(x(2)%p)

!ERROR: name in NULLIFY statement must have the POINTER attribute
Nullify(pi)

!ERROR: name in NULLIFY statement must have the POINTER attribute
Nullify(prp)

!ERROR: name in NULLIFY statement must be a variable or procedure pointer name
Nullify(maxvalue)

End Program

! Make sure that the compiler doesn't crash when NULLIFY is used in a context
! that has reported errors
module badNullify
  interface
    module function ptrFun()
      integer, pointer :: ptrFun
    end function
  end interface
contains
  !ERROR: 'ptrfun' was not declared a separate module procedure
  module function ptrFun()
    integer, pointer :: ptrFun
    real :: realVar
    nullify(ptrFun)
    nullify(realVar)
  end function
end module
