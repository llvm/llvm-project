! RUN: %python %S/test_folding.py %s %flang_fc1
! Test folding of LBOUND and UBOUND

module m
  real :: a3(42:52)
  real :: empty(52:42, 2:3, 10:1)
  integer, parameter :: lba3(*) = lbound(a3)
  logical, parameter :: test_lba3 = all(lba3 == [42])
  type :: t
    real :: a
  end type
  type(t) :: ta(0:2)
  character(len=2) :: ca(-1:1)
  interface
    function foo()
      real :: foo(2:3,4:6)
    end function
  end interface
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

  integer, parameter :: lbs_empty(*) = lbound(empty)
  logical, parameter :: test_lbs_empty = all(lbs_empty == [1, 2, 1])
  integer, parameter :: ubs_empty(*) = ubound(empty)
  logical, parameter :: test_ubs_empty = all(ubs_empty == [0, 3, 0])
  logical, parameter :: test_lb_empty_dim = lbound(empty, 1) == 1
  logical, parameter :: test_ub_empty_dim = ubound(empty, 1) == 0
 contains
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
  subroutine test3_bound_parameter
    ! Test [ul]bound with parameter arrays
    integer, parameter :: a1(1) = 0
    integer, parameter :: lba1(*) = lbound(a1)
    logical, parameter :: test_lba1 = all(lba1 == [1])
    integer, parameter :: uba1(*) = ubound(a1)
    logical, parameter :: test_uba1 = all(lba1 == [1])

    integer, parameter :: a2(0:1) = 0
    integer, parameter :: lba2(*) = lbound(a2)
    logical, parameter :: test_lba2 = all(lba2 == [0])
    integer, parameter :: uba2(*) = ubound(a2)
    logical, parameter :: test_uba2 = all(uba2 == [1])

    integer, parameter :: a3(-10:-5,1,4:6) = 0
    integer, parameter :: lba3(*) = lbound(a3)
    logical, parameter :: test_lba3 = all(lba3 == [-10, 1, 4])
    integer, parameter :: uba3(*) = ubound(a3)
    logical, parameter :: test_uba3 = all(uba3 == [-5, 1, 6])

    ! Exercise with DIM=
    logical, parameter :: test_lba3_dim = lbound(a3, 1) == -10 .and. &
         lbound(a3, 2) == 1 .and. &
         lbound(a3, 3) == 4
    logical, parameter :: test_uba3_dim = ubound(a3, 1) == -5 .and. &
         ubound(a3, 2) == 1 .and. &
         ubound(a3, 3) == 6
  end subroutine
  subroutine test4_bound_parentheses
    ! Test [ul]bound with (x) expressions
    integer :: a1(1) = 0
    logical, parameter :: test_lba1 = all(lbound((a1)) == [1])
    logical, parameter :: test_uba1 = all(ubound((a1)) == [1])
    integer :: a2(0:2) = 0
    logical, parameter :: test_lba2 = all(lbound((a2)) == [1])
    logical, parameter :: test_uba2 = all(ubound((a2)) == [3])
    integer :: a3(-1:0) = 0
    logical, parameter :: test_lba3 = all(lbound((a3)) == [1])
    logical, parameter :: test_uba3 = all(ubound((a3)) == [2])
    integer :: a4(-5:-1, 2:5) = 0
    logical, parameter :: test_lba4 = all(lbound((a4)) == [1, 1])
    logical, parameter :: test_uba4 = all(ubound((a4)) == [5, 4])

    ! Exercise with DIM=
    logical, parameter :: test_lba4_dim = lbound((a4), 1) == 1 .and. &
         lbound((a4), 2) == 1
    logical, parameter :: test_uba4_dim = ubound((a4), 1) == 5 .and. &
         ubound((a4), 2) == 4

    ! Exercise with parameter types
    integer, parameter :: pa1(1) = 0
    logical, parameter :: test_lbpa1 = all(lbound((pa1)) == [1])
    logical, parameter :: test_ubpa1 = all(ubound((pa1)) == [1])
    integer, parameter :: pa2(0:2) = 0
    logical, parameter :: test_lbpa2 = all(lbound((pa2)) == [1])
    logical, parameter :: test_ubpa2 = all(ubound((pa2)) == [3])
    integer, parameter :: pa3(-1:0) = 0
    logical, parameter :: test_lbpa3 = all(lbound((pa3)) == [1])
    logical, parameter :: test_ubpa3 = all(ubound((pa3)) == [2])
    integer, parameter :: pa4(-5:-1, 2:5) = 0
    logical, parameter :: test_lbpa4 = all(lbound((pa4)) == [1, 1])
    logical, parameter :: test_ubpa4 = all(ubound((pa4)) == [5, 4])

    ! Exercise with DIM=
    logical, parameter :: test_lbpa4_dim = lbound((pa4), 1) == 1 .and. &
         lbound((pa4), 2) == 1
    logical, parameter :: test_ubpa4_dim = ubound((pa4), 1) == 5 .and. &
         ubound((pa4), 2) == 4
  end
  subroutine test5_max_ubound
    ! Test maximum ubound value
    integer(8), parameter :: I64_MAX = INT(z'7fffffffffffffff', kind=8)
    integer, parameter :: a5(I64_MAX - 2 : I64_MAX) = [1, 2, 3]
    logical, parameter :: test_uba5 = ubound(a5, 1, kind=8) == I64_MAX
  end subroutine
end
