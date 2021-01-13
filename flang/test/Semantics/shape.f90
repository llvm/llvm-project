! RUN: %S/test_errors.sh %s %t %f18
! Test comparisons that use the intrinsic SHAPE() as an operand
program testShape
contains
  subroutine sub1(arrayDummy)
    integer :: arrayDummy(:)
    integer, allocatable :: arrayDeferred(:)
    integer :: arrayLocal(2) = [88, 99]
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
  end subroutine sub1
end program testShape
