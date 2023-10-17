! RUN: %python %S/test_errors.py %s %flang_fc1
! Test warnings and errors about DIM= arguments to transformational intrinsics

module m
 contains
  function f0a(a)
    real, intent(in) :: a(:)
    !ERROR: The value of DIM= (-1) may not be less than 1
    f0a = sum(a,dim=-1)
  end function
  function f0b(a)
    real, intent(in) :: a(:)
    !ERROR: The value of DIM= (2) may not be greater than 1
    f0b = sum(a,dim=2)
  end function
  function f1(a,d)
    real, intent(in) :: a(:)
    integer, optional, intent(in) :: d
    !PORTABILITY: The actual argument for DIM= is optional, pointer, or allocatable, and it is assumed to be present and equal to 1 at execution time
    f1 = sum(a,dim=d)
    !PORTABILITY: The actual argument for DIM= is optional, pointer, or allocatable, and it is assumed to be present and equal to 1 at execution time
    f1 = norm2(a,dim=d)
  end function
  function f2(a,d)
    real, intent(in) :: a(:)
    integer, pointer, intent(in) :: d
    !PORTABILITY: The actual argument for DIM= is optional, pointer, or allocatable, and it is assumed to be present and equal to 1 at execution time
    f2 = sum(a,dim=d)
  end function
  function f3(a,d)
    real, intent(in) :: a(:)
    integer, allocatable, intent(in) :: d
    !PORTABILITY: The actual argument for DIM= is optional, pointer, or allocatable, and it is assumed to be present and equal to 1 at execution time
    f3 = sum(a,dim=d)
  end function
  function f10a(a)
    real, intent(in) :: a(:,:)
    real, allocatable :: f10a(:)
    !ERROR: The value of DIM= (-1) may not be less than 1
    f10a = sum(a,dim=-1)
  end function
  function f10b(a)
    real, intent(in) :: a(:,:)
    real, allocatable :: f10b(:)
    !ERROR: The value of DIM= (3) may not be greater than 2
    f10b = sum(a,dim=3)
  end function
  function f11(a,d)
    real, intent(in) :: a(:,:)
    integer, optional, intent(in) :: d
    real, allocatable :: f11(:)
    !WARNING: The actual argument for DIM= is optional, pointer, or allocatable, and may not be absent during execution; parenthesize to silence this warning
    f11 = sum(a,dim=d)
    !WARNING: The actual argument for DIM= is optional, pointer, or allocatable, and may not be absent during execution; parenthesize to silence this warning
    f11 = norm2(a,dim=d)
  end function
  function f12(a,d)
    real, intent(in) :: a(:,:)
    integer, pointer, intent(in) :: d
    real, allocatable :: f12(:)
    !WARNING: The actual argument for DIM= is optional, pointer, or allocatable, and may not be absent during execution; parenthesize to silence this warning
    f12 = sum(a,dim=d)
  end function
  function f13(a,d)
    real, intent(in) :: a(:,:)
    integer, allocatable, intent(in) :: d
    real, allocatable :: f13(:)
    !WARNING: The actual argument for DIM= is optional, pointer, or allocatable, and may not be absent during execution; parenthesize to silence this warning
    f13 = sum(a,dim=d)
  end function
end module
