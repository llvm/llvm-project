! RUN: %python %S/test_errors.py %s %flang_fc1
! Test instantiation of components that are procedure pointers.
! 
program test
  type dtype(kindParam)
    integer, kind :: kindParam = 4
    !ERROR: KIND parameter value (66) of intrinsic type REAL did not resolve to a supported value
    !ERROR: KIND parameter value (55) of intrinsic type REAL did not resolve to a supported value
    procedure (real(kindParam)), pointer, nopass :: field => null()
  end type

  type base(kindParam)
    integer, kind :: kindParam = 4
    !ERROR: KIND parameter value (77) of intrinsic type REAL did not resolve to a supported value
    procedure (real(kindParam)), pointer, nopass :: field => null()
  end type
  type dependentType(kindParam)
    integer, kind :: kindParam = 4
    procedure (type(base(kindParam))), pointer, nopass :: field => null()
  end type

  ! OK unless entities are declared with the default type
  type badDefaultType(kindParam)
    integer, kind :: kindParam = 99
    !ERROR: KIND parameter value (99) of intrinsic type REAL did not resolve to a supported value
    !ERROR: KIND parameter value (44) of intrinsic type REAL did not resolve to a supported value
    procedure (real(kindParam)), pointer, nopass :: field => null()
  end type

  type parent(kindParam)
    integer, kind :: kindParam = 4
    !ERROR: KIND parameter value (33) of intrinsic type REAL did not resolve to a supported value
    !ERROR: KIND parameter value (88) of intrinsic type REAL did not resolve to a supported value
    procedure (real(kindParam)), pointer, nopass :: parentField => null()
  end type
  type, extends(parent) :: child
    integer :: field
  end type child
contains
  subroutine testGoodDefault(arg)
    type(dtype) :: arg
    if (associated(arg%field)) stop 'fail'
  end subroutine testGoodDefault

  subroutine testStar(arg)
    !ERROR: Value of KIND type parameter 'kindparam' must be constant
    type(dtype(*)),intent(inout) :: arg
    if (associated(arg%field)) stop 'fail'
  end subroutine testStar

  subroutine testBadDeclaration(arg)
    type(dtype(66)) :: arg
    if (associated(arg%field)) stop 'fail'
  end subroutine testBadDeclaration

  subroutine testBadLocalDeclaration()
    type(dtype(55)) :: local
    if (associated(local%field)) stop 'fail'
  end subroutine testBadLocalDeclaration

  subroutine testDependent()
    type(dependentType(77)) :: local
  end subroutine testDependent

  subroutine testBadDefault()
    type(badDefaultType) :: local
  end subroutine testBadDefault

  subroutine testBadDefaultWithBadDeclaration()
    type(badDefaultType(44)) :: local
  end subroutine testBadDefaultWithBadDeclaration

  subroutine testBadDefaultWithGoodDeclaration()
    type(badDefaultType(4)) :: local
  end subroutine testBadDefaultWithGoodDeclaration

  subroutine testExtended()
    type(child(33)) :: local1
    type(child(4)) :: local2
    type(parent(88)) :: local3
    type(parent(8)) :: local4
  end subroutine testExtended
end program test
