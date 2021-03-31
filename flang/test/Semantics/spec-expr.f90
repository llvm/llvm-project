! RUN: %S/test_errors.sh %s %t %f18
! Tests for the 14 items that specify a "specification expression" in section
! 10.1.11

! a constant or subobject of a constant,
subroutine s1()
  type dType
    integer :: field
  end type dType

  type(dType), parameter :: dConst = dType(3)
  real, dimension(3) :: realVar1
  real, dimension(dConst%field) :: realVar2
end subroutine s1

! an object designator with a base object that is a dummy argument that has 
! neither the OPTIONAL nor the INTENT (OUT) attribute,
subroutine s2(inArg, inoutArg, outArg, optArg)
  integer, intent(in) :: inArg
  integer, intent(inout) :: inoutArg
  integer, intent(out) :: outArg
  integer, intent(in), optional :: optArg
  real, dimension(inArg) :: realVar1
  real, dimension(inoutArg) :: realVar2
  !ERROR: Invalid specification expression: reference to INTENT(OUT) dummy argument 'outarg'
  real, dimension(outArg) :: realVar3
  !ERROR: Invalid specification expression: reference to OPTIONAL dummy argument 'optarg'
  real, dimension(optArg) :: realVar4

  outArg = 3
end subroutine s2

! an object designator with a base object that is in a common block,
subroutine s3()
  integer :: intVar
  common intCommonVar
  real, dimension(intCommonVar) :: realVar
end subroutine s3

! an object designator with a base object that is made accessible by
!    use or host association,
module m4
  integer :: intVar
end module m4

subroutine s4()
  use m4
  real, dimension(intVar) :: realVar
end subroutine s4

! an array constructor where each element and each scalar-int-expr of 
!   each ac-implied-do-control is a restricted expression,
subroutine s5()
  real, dimension(storage_size([1,2])) :: realVar
end subroutine s5

! a structure constructor where each component is a restricted expression,
subroutine s6()
  type :: dType
    integer :: field1
    integer :: field2
  end type dType

  real, dimension(storage_size(dType(1, 2))) :: realArray
end subroutine s6

! a specification inquiry where each designator or argument is
!   (a) a restricted expression or
subroutine s7a()
  real, dimension(3) :: realArray1
  real, dimension(size(realArray1)) :: realArray2
end subroutine s7a

! a specification inquiry where each designator or argument is
!   (b) a variable that is not an optional dummy argument, and whose
!     properties inquired about are not
!     (i)   dependent on the upper bound of the last dimension of an 
!       assumed-size array,
subroutine s7bi(assumedArg)
  integer, dimension(2, *) :: assumedArg
  real, dimension(ubound(assumedArg, 1)) :: realArray1
  !ERROR: DIM=2 dimension is out of range for rank-2 assumed-size array
  real, dimension(ubound(assumedArg, 2)) :: realArray2
end subroutine s7bi

! a specification inquiry where each designator or argument is
!   (b) a variable that is not an optional dummy argument, and whose
!     properties inquired about are not
!     (ii)  deferred, or
subroutine s7bii(dummy)
  character(len=:), pointer :: dummy
  ! Should be an error since "dummy" is deferred, but all compilers handle it
  real, dimension(len(dummy)) :: realArray
end subroutine s7bii

! a specification inquiry where each designator or argument is
!   (b) a variable that is not an optional dummy argument, and whose
!     properties inquired about are not
!  (iii) defined by an expression that is not a restricted expression,
subroutine s7biii()
  integer, parameter :: localConst = 5
  integer :: local = 5
  ! OK, since "localConst" is a constant
  real, dimension(localConst) :: realArray1
  !ERROR: Invalid specification expression: reference to local entity 'local'
  real, dimension(local) :: realArray2
end subroutine s7biii

! a specification inquiry that is a constant expression,
subroutine s8()
  integer :: iVar
  real, dimension(bit_size(iVar)) :: realArray
end subroutine s8

! a reference to the intrinsic function PRESENT,
subroutine s9(optArg)
  integer, optional :: optArg
  real, dimension(merge(3, 4, present(optArg))) :: realArray
end subroutine s9

! a reference to any other standard intrinsic function where each
!   argument is a restricted expression,
subroutine s10()
  integer :: iVar
  real, dimension(bit_size(iVar)) :: realArray
end subroutine s10

! a reference to a transformational function from the intrinsic module 
!   IEEE_ARITHMETIC, IEEE_EXCEPTIONS, or ISO_C_BINDING, where each argument 
!   is a restricted expression,
subroutine s11()
  use ieee_exceptions
  real, dimension(merge(3, 4, ieee_support_halting(ieee_invalid))) :: realArray
end subroutine s11

! a reference to a specification function where each argument is a 
!   restricted expression,
module m12
  contains
    pure function specFunc(arg)
      integer, intent(in) :: arg
      integer :: specFunc
      specFunc = 3 + arg
    end function specFunc
end module m12

subroutine s12()
  use m12
  real, dimension(specFunc(2)) :: realArray
end subroutine s12

! a type parameter of the derived type being defined,
subroutine s13()
  type :: dtype(param)
    integer, len :: param
    real, dimension(param) :: realField
  end type dtype
end subroutine s13

! an ac-do-variable within an array constructor where each 
!   scalar-int-expr of the corresponding ac-implied-do-control is a restricted 
!   expression, or
subroutine s14()
  real, dimension(5) :: realField = [(i, i = 1, 5)]
end subroutine s14

! a restricted expression enclosed in parentheses,where each subscript, 
!   section subscript, substring starting point, substring ending point, and 
!   type parameter value is a restricted expression
subroutine s15()
  type :: dtype(param)
    integer, len :: param
    real, dimension((param + 2)) :: realField
  end type dtype
end subroutine s15

! Regression test: don't get confused by host association
subroutine s16(n)
  integer :: n
 contains
  subroutine inner(r)
    real, dimension(n) :: r
  end subroutine
end subroutine s16
