! RUN: %python %S/test_errors.py %s %flang_fc1
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
!ERROR: 'p' may not appear in NULLIFY
!BECAUSE: 'p' is not a pointer
Nullify(x(2)%p)

!ERROR: 'pi' may not appear in NULLIFY
!BECAUSE: 'pi' is not a pointer
Nullify(pi)

!ERROR: 'prp' may not appear in NULLIFY
!BECAUSE: 'prp' is not a pointer
Nullify(prp)

!ERROR: 'maxvalue' may not appear in NULLIFY
!BECAUSE: 'maxvalue' is not a pointer
Nullify(maxvalue)

End Program

! Make sure that the compiler doesn't crash when NULLIFY is used in a context
! that has reported errors
module badNullify
  interface
    function ptrFun()
      integer, pointer :: ptrFun
    end function
  end interface
contains
  !ERROR: 'ptrfun' was not declared a separate module procedure
  !ERROR: 'ptrfun' is already declared in this scoping unit
  module function ptrFun()
    integer, pointer :: ptrFun
    real :: realVar
    nullify(ptrFun)
    !ERROR: 'realvar' may not appear in NULLIFY
    !BECAUSE: 'realvar' is not a pointer
    nullify(realVar)
  end function
end module
