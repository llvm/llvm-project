! RUN: %python %S/test_errors.py %s %flang_fc1
! Relational operators on CHARACTER operands require both operands to have
! the same kind.

module m
  character(*,kind=1), parameter :: pc1 = 1_'abc'
  character(*,kind=2), parameter :: pc2 = 2_'abc'
  character(*,kind=4), parameter :: pc4 = 4_'abc'
  logical, parameter :: same1 = pc1 == 1_'abc'
  logical, parameter :: same2 = pc2 < 2_'abd'
  logical, parameter :: same4 = pc4 >= 4_'abc'
  !ERROR: CHARACTER operands do not have same KIND
  logical, parameter :: diff14 = pc1 == pc4
contains
  subroutine same_kind(a1, b1, a2, b2, a4, b4, arr1, arr4)
    character(4,kind=1) :: a1
    character(5,kind=1) :: b1
    character(4,kind=2) :: a2
    character(5,kind=2) :: b2
    character(4,kind=4) :: a4
    character(5,kind=4) :: b4
    character(4,kind=1) :: arr1(3)
    character(4,kind=4) :: arr4(3)
    logical :: l, la(3)
    l = a1 == b1
    l = a1 /= b1
    l = a2 < b2
    l = a2 <= b2
    l = a4 > b4
    l = a4 >= b4
    l = a1 .eq. 'xyz'
    l = a4 .lt. 4_'xyz'
    la = arr1 == a1
    la = arr4 > arr4
  end

  subroutine different_kind(a1, a2, a4, arr1, arr4)
    character(4,kind=1) :: a1
    character(4,kind=2) :: a2
    character(4,kind=4) :: a4
    character(4,kind=1) :: arr1(3)
    character(4,kind=4) :: arr4(3)
    logical :: l, la(3)
    !ERROR: CHARACTER operands do not have same KIND
    l = a1 == a2
    !ERROR: CHARACTER operands do not have same KIND
    l = a2 /= a1
    !ERROR: CHARACTER operands do not have same KIND
    l = a1 < a4
    !ERROR: CHARACTER operands do not have same KIND
    l = a4 <= a1
    !ERROR: CHARACTER operands do not have same KIND
    l = a2 > a4
    !ERROR: CHARACTER operands do not have same KIND
    l = a4 >= a2
    !ERROR: CHARACTER operands do not have same KIND
    l = a1 .eq. 4_'xyz'
    !ERROR: CHARACTER operands do not have same KIND
    l = 2_'xyz' .ne. a4
    !ERROR: CHARACTER operands do not have same KIND
    l = 1_'abc' < 4_'abc'
    !ERROR: CHARACTER operands do not have same KIND
    l = pc2 == pc4
    !ERROR: CHARACTER operands do not have same KIND
    la = arr1 == arr4
    !ERROR: CHARACTER operands do not have same KIND
    la = arr4 > a1
  end
end
