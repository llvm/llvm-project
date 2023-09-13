! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  use iso_c_binding
  type haslen(L)
    integer, len :: L
  end type
 contains
  subroutine test(assumedType, poly, nclen)
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
    character(2), target :: ch
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
    cp = c_loc(ch(1:1)) ! ok)
    !ERROR: PRIVATE name '__address' is only accessible within module '__fortran_builtins'
    cp = c_ptr(0)
    !ERROR: PRIVATE name '__address' is only accessible within module '__fortran_builtins'
    cfp = c_funptr(0)
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(c_ptr) and TYPE(c_funptr)
    cp = cfp
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(c_funptr) and TYPE(c_ptr)
    cfp = cp
  end
end module
