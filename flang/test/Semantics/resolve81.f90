! RUN: %python %S/test_errors.py %s %flang_fc1
! C801 The same attr-spec shall not appear more than once in a given
! type-declaration-stmt.
!
! R801 type-declaration-stmt ->
!        declaration-type-spec [[, attr-spec]... ::] entity-decl-list
!  attr-spec values are:
!    PUBLIC, PRIVATE, ALLOCATABLE, ASYNCHRONOUS, CODIMENSION, CONTIGUOUS,
!    DIMENSION (array-spec), EXTERNAL, INTENT (intent-spec), INTRINSIC,
!    BIND(C), OPTIONAL, PARAMETER, POINTER, PROTECTED, SAVE, TARGET, VALUE,
!    VOLATILE
module m

  !WARNING: Attribute 'PUBLIC' cannot be used more than once
  real, public, allocatable, public :: publicVar
  !WARNING: Attribute 'PRIVATE' cannot be used more than once
  real, private, allocatable, private :: privateVar
  !WARNING: Attribute 'ALLOCATABLE' cannot be used more than once
  real, allocatable, allocatable :: allocVar
  !WARNING: Attribute 'ASYNCHRONOUS' cannot be used more than once
  real, asynchronous, public, asynchronous :: asynchVar
  !ERROR: Attribute 'CODIMENSION' cannot be used more than once
  real, codimension[*], codimension[*] :: codimensionVar
  !WARNING: Attribute 'CONTIGUOUS' cannot be used more than once
  real, contiguous, pointer, contiguous :: contigVar(:)
  !ERROR: Attribute 'DIMENSION' cannot be used more than once
  real, dimension(5), dimension(5) :: arrayVar
  !WARNING: Attribute 'EXTERNAL' cannot be used more than once
  real, external, external :: externFunc
  !WARNING: Attribute 'INTRINSIC' cannot be used more than once
  !ERROR: 'cos' may not have both the BIND(C) and INTRINSIC attributes
  !ERROR: An interface name with the BIND attribute must appear if the BIND attribute appears in a procedure declaration
  real, intrinsic, bind(c), intrinsic :: cos
  !WARNING: Attribute 'BIND(C)' cannot be used more than once
  integer, bind(c), volatile, bind(c) :: bindVar
  !WARNING: Attribute 'PARAMETER' cannot be used more than once
  real, parameter, parameter :: realConst = 4.3
  !WARNING: Attribute 'POINTER' cannot be used more than once
  real, pointer, pointer :: realPtr
  !WARNING: Attribute 'PROTECTED' cannot be used more than once
  real, protected, protected :: realProt
  !WARNING: Attribute 'SAVE' cannot be used more than once
  real, save, save :: saveVar
  !WARNING: Attribute 'TARGET' cannot be used more than once
  real, target, target :: targetVar
  !WARNING: Attribute 'VOLATILE' cannot be used more than once
  real, volatile, volatile :: volatileVar

contains
    subroutine testTypeDecl(arg1, arg2, arg3, arg4, arg5, arg6)
      !WARNING: Attribute 'INTENT(IN)' cannot be used more than once
      real, intent(in), intent(in) :: arg1
      !WARNING: Attribute 'INTENT(OUT)' cannot be used more than once
      real, intent(out), intent(out) :: arg2
      !WARNING: Attribute 'INTENT(INOUT)' cannot be used more than once
      real, intent(inout), intent(inout) :: arg3
      !WARNING: Attribute 'OPTIONAL' cannot be used more than once
      integer, optional, intent(in), optional :: arg4
      !WARNING: Attribute 'VALUE' cannot be used more than once
      integer, value, intent(in), value :: arg5
      !ERROR: Attributes 'INTENT(IN)' and 'INTENT(INOUT)' conflict with each other
      integer, intent(in), pointer, intent(inout) :: arg6

      arg2 =3.5
    end subroutine testTypeDecl
end module m
