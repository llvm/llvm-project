! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Procedure pointer assignments and argument association with intrinsic functions
program test
  abstract interface
    real function realToReal(a)
      real, intent(in) :: a
    end function
    real function intToReal(n)
      integer, intent(in) :: n
    end function
  end interface
  procedure(), pointer :: noInterfaceProcPtr
  procedure(realToReal), pointer :: realToRealProcPtr
  procedure(intToReal), pointer :: intToRealProcPtr
  intrinsic :: float ! restricted specific intrinsic functions
  intrinsic :: sqrt ! unrestricted specific intrinsic functions
  external :: noInterfaceExternal
  interface
    elemental real function userElemental(a)
      real, intent(in) :: a
    end function
  end interface

  !ERROR: 'float' is not an unrestricted specific intrinsic procedure
  noInterfaceProcPtr => float
  !ERROR: 'float' is not an unrestricted specific intrinsic procedure
  intToRealProcPtr => float
  !ERROR: 'float' is not an unrestricted specific intrinsic procedure
  call sub1(float)
  !ERROR: 'float' is not an unrestricted specific intrinsic procedure
  call sub2(float)
  !ERROR: 'float' is not an unrestricted specific intrinsic procedure
  call sub3(float)

  noInterfaceProcPtr => sqrt ! ok
  realToRealProcPtr => sqrt ! ok
  !ERROR: Procedure pointer 'inttorealprocptr' associated with incompatible procedure designator 'sqrt': incompatible dummy argument #1: incompatible dummy data object types: REAL(4) vs INTEGER(4)
  intToRealProcPtr => sqrt
  call sub1(sqrt) ! ok
  call sub2(sqrt) ! ok
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'p=': incompatible dummy argument #1: incompatible dummy data object types: REAL(4) vs INTEGER(4)
  call sub3(sqrt)

  print *, implicitExtFunc()
  call implicitExtSubr
  noInterfaceProcPtr => implicitExtFunc ! ok
  noInterfaceProcPtr => implicitExtSubr ! ok
  noInterfaceProcPtr => noInterfaceExternal ! ok
  realToRealProcPtr => noInterfaceExternal ! ok
  intToRealProcPtr => noInterfaceExternal !ok
  call sub1(noInterfaceExternal) ! ok
  !WARNING: Actual procedure argument has an implicit interface which is not known to be compatible with dummy argument 'p=' which has an explicit interface
  call sub2(noInterfaceExternal)
  !WARNING: Actual procedure argument has an implicit interface which is not known to be compatible with dummy argument 'p=' which has an explicit interface
  call sub3(noInterfaceExternal)

  !ERROR: Procedure pointer 'nointerfaceprocptr' with implicit interface may not be associated with procedure designator 'userelemental' with explicit interface that cannot be called via an implicit interface
  noInterfaceProcPtr => userElemental
  !ERROR: Non-intrinsic ELEMENTAL procedure 'userelemental' may not be passed as an actual argument
  call sub1(userElemental)

 contains
  subroutine sub1(p)
    external :: p
  end subroutine
  subroutine sub2(p)
    procedure(realToReal) :: p
  end subroutine
  subroutine sub3(p)
    procedure(intToReal) :: p
  end subroutine
end
