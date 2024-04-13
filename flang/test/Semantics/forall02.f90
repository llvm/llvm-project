! RUN: %python %S/test_errors.py %s %flang_fc1

module m1
  type :: impureFinal
  contains
    final :: impureSub
    final :: impureSubRank1
    final :: impureSubRank2
  end type

 contains

  impure subroutine impureSub(x)
    type(impureFinal), intent(in) :: x
  end subroutine

  impure subroutine impureSubRank1(x)
    type(impureFinal), intent(in) :: x(:)
  end subroutine

  impure subroutine impureSubRank2(x)
    type(impureFinal), intent(in) :: x(:,:)
  end subroutine

  subroutine s1()
    implicit none
    integer :: i
    type(impureFinal), allocatable :: ifVar, ifvar1
    type(impureFinal), allocatable :: ifArr1(:), ifArr2(:,:)
    type(impureFinal) :: if0
    integer a(10)
    allocate(ifVar)
    allocate(ifVar1)
    allocate(ifArr1(5), ifArr2(5,5))

    ! Error to invoke an IMPURE FINAL procedure in a FORALL
    forall (i = 1:10)
      !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
      !ERROR: Impure procedure 'impuresub' is referenced by finalization in a FORALL
      ifvar = ifvar1
    end forall

    forall (i = 1:5)
      !ERROR: Impure procedure 'impuresub' is referenced by finalization in a FORALL
      ifArr1(i) = if0
    end forall

    forall (i = 1:5)
      !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
      !ERROR: Impure procedure 'impuresubrank1' is referenced by finalization in a FORALL
      ifArr1 = if0
    end forall

    forall (i = 1:5)
      !ERROR: Impure procedure 'impuresubrank1' is referenced by finalization in a FORALL
      ifArr2(i,:) = if0
    end forall

    forall (i = 1:5)
      !WARNING: FORALL index variable 'i' not used on left-hand side of assignment
      !ERROR: Impure procedure 'impuresubrank2' is referenced by finalization in a FORALL
      ifArr2(:,:) = if0
    end forall
  end subroutine

end module m1

