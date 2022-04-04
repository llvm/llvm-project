! RUN: %python %S/test_errors.py %s %flang_fc1
! Interfaces are allowed to extend intrinsic procedures, with limitations
module m1
  intrinsic sin
  interface sin
    module procedure :: charcpy
  end interface
  interface cos ! no INTRINSIC statement
    module procedure :: charcpy
  end interface
  intrinsic mvbits
  interface mvbits
    module procedure :: negate
  end interface
  interface move_alloc ! no INTRINSIC statement
    module procedure :: negate
  end interface
  interface tan ! not explicitly INTRINSIC
    module procedure :: negate ! a subroutine
  end interface
  interface acos
    module procedure :: minus ! override
  end interface
  intrinsic atan
  !ERROR: Generic interface 'atan' with explicit intrinsic function of the same name may not have specific procedure 'negate' that is a subroutine
  interface atan
    module procedure :: negate ! a subroutine
  end interface
 contains
  character function charcpy(x)
    character, intent(in) :: x
    charcpy = x
  end function
  subroutine negate(x)
    real, intent(in out) :: x
    x = -x
  end subroutine
  real elemental function minus(x)
    real, intent(in) :: x
    minus = -x
  end function
  subroutine test
    integer, allocatable :: j, k
    real :: x
    character :: str
    x = sin(x)
    str = sin(str) ! charcpy
    x = cos(x)
    str = cos(str) ! charcpy
    call mvbits(j,0,1,k,0)
    call mvbits(x) ! negate
    call move_alloc(j, k)
    call move_alloc(x) ! negate
    !ERROR: Cannot call subroutine 'tan' like a function
    x = tan(x)
    x = acos(x) ! user's interface overrides intrinsic
  end subroutine
end module
