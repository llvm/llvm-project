! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for UNIT=function()
module m1
  integer, target :: itarget
  character(20), target :: ctarget
  logical, target :: ltarget
  interface gf
    module procedure :: intf, pintf, pchf, logf, plogf
  end interface
 contains
  integer function intf(n)
    integer(1), intent(in) :: n
    intf = n
  end function
  function pintf(n)
    integer(2), intent(in) :: n
    integer, pointer :: pintf
    pintf => itarget
    pintf = n
  end function
  function pchf(n)
    integer(4), intent(in) :: n
    character(:), pointer :: pchf
    pchf => ctarget
  end function
  logical function logf(n)
    integer(8), intent(in) :: n
    logf = .true.
  end function
  function plogf(n)
    integer(16), intent(in) :: n
    logical, pointer :: plf
    plf => ltarget
  end function
  subroutine test
    write(intf(6_1),"('hi')")
    write(pintf(6_2),"('hi')")
    write(pchf(123_4),"('hi')")
    write(gf(6_1),"('hi')")
    write(gf(6_2),"('hi')")
    write(gf(666_4),"('hi')")
    !ERROR: I/O unit must be a character variable or a scalar integer expression
    write(logf(666_8),"('hi')")
    !ERROR: I/O unit must be a character variable or a scalar integer expression
    write(plogf(666_16),"('hi')")
    !ERROR: I/O unit must be a character variable or a scalar integer expression
    write(gf(666_8),"('hi')")
    !ERROR: I/O unit must be a character variable or a scalar integer expression
    write(gf(666_16),"('hi')")
    !ERROR: I/O unit must be a character variable or a scalar integer expression
    write(null(),"('hi')")
  end subroutine
end module
