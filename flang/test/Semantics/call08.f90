! RUN: %python %S/test_errors.py %s %flang_fc1
! Test 15.5.2.8 coarray dummy arguments

module m

  real :: c1[*]
  real, volatile :: c2[*]

 contains

  subroutine s01(x)
    real :: x[*]
  end subroutine
  subroutine s02(x)
    real, volatile :: x[*]
  end subroutine
  subroutine s03(x)
    real, contiguous :: x(:)[*]
  end subroutine
  subroutine s04(x)
    real :: x(*)[*]
  end subroutine

  subroutine test(x,c3,c4)
    real :: scalar
    real :: x(:)[*]
    real, intent(in) :: c3(:)[*]
    real, contiguous, intent(in) :: c4(:)[*]
    character(2) :: coarr(2)[*] = [ "ab", "cd" ]
    call s01(c1) ! ok
    call s02(c2) ! ok
    call s03(c4) ! ok
    call s04(c4) ! ok
    !ERROR: Actual argument associated with coarray dummy argument 'x=' must be a coarray
    call s01(scalar)
    !ERROR: VOLATILE coarray may not be associated with non-VOLATILE coarray dummy argument 'x='
    call s01(c2)
    !ERROR: non-VOLATILE coarray may not be associated with VOLATILE coarray dummy argument 'x='
    call s02(c1)
    !ERROR: Actual argument associated with a CONTIGUOUS coarray dummy argument 'x=' must be simply contiguous
    call s03(c3)
    !ERROR: Actual argument associated with a CONTIGUOUS coarray dummy argument 'x=' must be simply contiguous
    call s03(x)
    !ERROR: Actual argument associated with coarray dummy argument 'x=' (not assumed shape or rank) must be simply contiguous
    call s04(c3)
    !ERROR: Actual argument associated with coarray dummy argument 'x=' (not assumed shape or rank) must be simply contiguous
    call s04(x)
    print *, ichar(coarr(:)(1:1)) ! ok, ensure no bogus contiguity error
  end subroutine
end module
