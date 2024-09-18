! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
module m
  use iso_c_binding
  type haslen(L)
    integer, len :: L
  end type
  integer, target :: targ
 contains
  subroutine subr
  end
  subroutine test(assumedType, poly, nclen, n)
    type(*), target :: assumedType
    class(*), target ::  poly
    type(c_ptr) cp
    type(c_funptr) cfp
    real notATarget
    !PORTABILITY: Procedure pointer 'pptr' should not have an ELEMENTAL intrinsic as its interface
    procedure(sin), pointer :: pptr
    real, target :: arr(3)
    type(hasLen(1)), target :: clen
    type(hasLen(*)), target :: nclen
    integer, intent(in) :: n
    character(2), target :: ch
    real :: arr1(purefun1(c_loc(targ))) ! ok
    real :: arr2(purefun2(c_funloc(subr))) ! ok
    character(:), allocatable, target :: deferred
    character(n), pointer :: p2ch
    !ERROR: C_LOC() argument must be a data pointer or target
    cp = c_loc(notATarget)
    !ERROR: C_LOC() argument must be a data pointer or target
    cp = c_loc(pptr)
    !ERROR: C_LOC() argument must be contiguous
    cp = c_loc(arr(1:3:2))
    !ERROR: C_LOC() argument may not be a zero-sized array
    cp = c_loc(arr(3:1))
    !ERROR: C_LOC() argument must have an intrinsic type, assumed type, or non-polymorphic derived type with no non-constant length parameter
    cp = c_loc(poly)
    cp = c_loc(clen) ! ok
    !ERROR: C_LOC() argument must have an intrinsic type, assumed type, or non-polymorphic derived type with no non-constant length parameter
    cp = c_loc(nclen)
    !ERROR: C_LOC() argument may not be zero-length character
    cp = c_loc(ch(2:1))
    !WARNING: C_LOC() argument has non-interoperable intrinsic type, kind, or length
    cp = c_loc(ch)
    cp = c_loc(ch(1:1)) ! ok
    cp = c_loc(deferred) ! ok
    cp = c_loc(p2ch) ! ok
    !ERROR: PRIVATE name '__address' is only accessible within module '__fortran_builtins'
    cp = c_ptr(0)
    !ERROR: PRIVATE name '__address' is only accessible within module '__fortran_builtins'
    cfp = c_funptr(0)
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(c_ptr) and TYPE(c_funptr)
    cp = cfp
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(c_funptr) and TYPE(c_ptr)
    cfp = cp
  end
  pure integer function purefun1(p)
    type(c_ptr), intent(in) :: p
    purefun1 = 1
  end
  pure integer function purefun2(p)
    type(c_funptr), intent(in) :: p
    purefun2 = 1
  end
end module
