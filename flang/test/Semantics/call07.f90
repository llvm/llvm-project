! RUN: %python %S/test_errors.py %s %flang_fc1
! Test 15.5.2.7 constraints and restrictions for POINTER dummy arguments.

module m
  real :: coarray(10)[*]
 contains

  subroutine s01(p)
    real, pointer, contiguous, intent(in) :: p(:)
  end subroutine
  subroutine s02(p)
    real, pointer :: p(:)
  end subroutine
  subroutine s03(p)
    real, pointer, intent(in) :: p(:)
  end subroutine
  subroutine s04(p)
    real, pointer :: p
  end subroutine

  subroutine test
    !ERROR: CONTIGUOUS POINTER must be an array
    real, pointer, contiguous :: a01 ! C830
    real, pointer :: a02(:)
    real, target :: a03(10)
    real :: a04(10) ! not TARGET
    call s01(a03) ! ok
    !WARNING: Target of CONTIGUOUS pointer association is not known to be contiguous
    call s01(a02)
    !ERROR: CONTIGUOUS pointer may not be associated with a discontiguous target
    call s01(a03(::2))
    call s02(a02) ! ok
    call s03(a03) ! ok
    !ERROR: Actual argument associated with POINTER dummy argument 'p=' must also be POINTER unless INTENT(IN)
    call s02(a03)
    !ERROR: Actual argument associated with POINTER dummy argument 'p=' must also be POINTER unless INTENT(IN)
    call s04(a02(1))
    !ERROR: An array section with a vector subscript may not be a pointer target
    call s03(a03([1,2,4]))
    !ERROR: A coindexed object may not be a pointer target
    call s03(coarray(:)[1])
    !ERROR: Target associated with dummy argument 'p=' must be a designator or a call to a pointer-valued function
    call s03([1.])
    !ERROR: In assignment to object dummy argument 'p=', the target 'a04' is not an object with POINTER or TARGET attributes
    call s03(a04)
  end subroutine
end module
