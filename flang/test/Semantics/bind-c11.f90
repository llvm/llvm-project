! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  !ERROR: A scalar interoperable variable may not be ALLOCATABLE or POINTER
  real, allocatable, bind(c) :: x1
  !ERROR: A scalar interoperable variable may not be ALLOCATABLE or POINTER
  real, pointer, bind(c) :: x2
  !ERROR: BIND(C) array must have explicit shape or be assumed-size unless a dummy argument without the VALUE attribute
  real, allocatable, bind(c) :: x3(:)
 contains
  subroutine s1(x) bind(c)
    !ERROR: A BIND(C) VALUE dummy argument must have an interoperable type
    logical(2), intent(in), value :: x
  end
  subroutine s2(x) bind(c)
    !PORTABILITY: An interoperable procedure with an OPTIONAL dummy argument might not be portable
    integer, intent(in), optional :: x
  end
  !ERROR: A subprogram interface with the BIND attribute may not have an alternate return argument
  subroutine s3(*) bind(c)
  end
end
