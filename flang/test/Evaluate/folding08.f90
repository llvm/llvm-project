! RUN: %python %S/test_folding.py %s %flang_fc1
! Test folding of LBOUND and UBOUND

module m
  real :: a3(42:52)
  integer, parameter :: lba3(*) = lbound(a3)
  logical, parameter :: test_lba3 = all(lba3 == [42])
  type :: t
    real :: a
  end type
  type(t) :: ta(0:2)
  character(len=2) :: ca(-1:1)
  integer, parameter :: lbtadim = lbound(ta,1)
  logical, parameter :: test_lbtadim = lbtadim == 0
  integer, parameter :: ubtadim = ubound(ta,1)
  logical, parameter :: test_ubtadim = ubtadim == 2
  integer, parameter :: lbta1(*) = lbound(ta)
  logical, parameter :: test_lbta1 = all(lbta1 == [0])
  integer, parameter :: ubta1(*) = ubound(ta)
  logical, parameter :: test_ubta1 = all(ubta1 == [2])
  integer, parameter :: lbta2(*) = lbound(ta(:))
  logical, parameter :: test_lbta2 = all(lbta2 == [1])
  integer, parameter :: ubta2(*) = ubound(ta(:))
  logical, parameter :: test_ubta2 = all(ubta2 == [3])
  integer, parameter :: lbta3(*) = lbound(ta%a)
  logical, parameter :: test_lbta3 = all(lbta3 == [1])
  integer, parameter :: ubta3(*) = ubound(ta%a)
  logical, parameter :: test_ubta3 = all(ubta3 == [3])
  integer, parameter :: lbca1(*) = lbound(ca)
  logical, parameter :: test_lbca1 = all(lbca1 == [-1])
  integer, parameter :: ubca1(*) = ubound(ca)
  logical, parameter :: test_ubca1 = all(ubca1 == [1])
  integer, parameter :: lbca2(*) = lbound(ca(:)(1:1))
  logical, parameter :: test_lbca2 = all(lbca2 == [1])
  integer, parameter :: ubca2(*) = ubound(ca(:)(1:1))
  logical, parameter :: test_ubca2 = all(ubca2 == [3])
  integer, parameter :: lbfoo(*) = lbound(foo())
  logical, parameter :: test_lbfoo = all(lbfoo == [1,1])
  integer, parameter :: ubfoo(*) = ubound(foo())
  logical, parameter :: test_ubfoo = all(ubfoo == [2,3])
 contains
  function foo()
    real :: foo(2:3,4:6)
  end function
  subroutine test(n1,a1,a2)
    integer, intent(in) :: n1
    real, intent(in) :: a1(1:n1), a2(0:*)
    integer, parameter :: lba1(*) = lbound(a1)
    logical, parameter :: test_lba1 = all(lba1 == [1])
    integer, parameter :: lba2(*) = lbound(a2)
    logical, parameter :: test_lba2 = all(lba2 == [0])
  end subroutine
  subroutine test2
    real :: a(2:3,4:6)
    associate (b => a)
      block
        integer, parameter :: lbb(*) = lbound(b)
        logical, parameter :: test_lbb = all(lbb == [2,4])
        integer, parameter :: ubb(*) = ubound(b)
        logical, parameter :: test_ubb = all(ubb == [3,6])
      end block
    end associate
    associate (b => a + 0)
      block
        integer, parameter :: lbb(*) = lbound(b)
        logical, parameter :: test_lbb = all(lbb == [1,1])
        integer, parameter :: ubb(*) = ubound(b)
        logical, parameter :: test_ubb = all(ubb == [2,3])
      end block
    end associate
  end subroutine
end
