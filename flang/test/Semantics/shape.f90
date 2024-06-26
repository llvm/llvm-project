! RUN: %python %S/test_errors.py %s %flang_fc1
! Test comparisons that use the intrinsic SHAPE() as an operand
program testShape
contains
  subroutine sub1(arrayDummy, assumedRank)
    integer :: arrayDummy(:), assumedRank(..)
    integer, allocatable :: arrayDeferred(:)
    integer :: arrayLocal(2) = [88, 99]
    integer, parameter :: aRrs = rank(shape(assumedRank))
    integer(kind=merge(kind(1),-1,aRrs == 1)) :: test_aRrs
    !ERROR: Dimension 1 of left operand has extent 1, but right operand has extent 0
    !ERROR: Dimension 1 of left operand has extent 1, but right operand has extent 0
    if (all(shape(arrayDummy)==shape(8))) then
      print *, "hello"
    end if
    !ERROR: Dimension 1 of left operand has extent 0, but right operand has extent 1
    !ERROR: Dimension 1 of left operand has extent 0, but right operand has extent 1
    if (all(shape(27)==shape(arrayDummy))) then
      print *, "hello"
    end if
    if (all(64==shape(arrayDummy))) then
      print *, "hello"
    end if
    !ERROR: Dimension 1 of left operand has extent 1, but right operand has extent 0
    !ERROR: Dimension 1 of left operand has extent 1, but right operand has extent 0
    if (all(shape(arrayDeferred)==shape(8))) then
      print *, "hello"
    end if
    !ERROR: Dimension 1 of left operand has extent 0, but right operand has extent 1
    !ERROR: Dimension 1 of left operand has extent 0, but right operand has extent 1
    if (all(shape(27)==shape(arrayDeferred))) then
      print *, "hello"
    end if
    if (all(64==shape(arrayDeferred))) then
      print *, "hello"
    end if
    !ERROR: Dimension 1 of left operand has extent 1, but right operand has extent 0
    !ERROR: Dimension 1 of left operand has extent 1, but right operand has extent 0
    if (all(shape(arrayLocal)==shape(8))) then
      print *, "hello"
    end if
    !ERROR: Dimension 1 of left operand has extent 0, but right operand has extent 1
    !ERROR: Dimension 1 of left operand has extent 0, but right operand has extent 1
    if (all(shape(27)==shape(arrayLocal))) then
      print *, "hello"
    end if
    if (all(64==shape(arrayLocal))) then
      print *, "hello"
    end if
    ! These can't be checked at compilation time
    if (any(shape(assumedRank) == [1])) stop
    if (any(lbound(assumedRank) == [1,2])) stop
    if (any(ubound(assumedRank) == [1,2,3])) stop
  end subroutine sub1
end program testShape
