! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for 8.6.4(1)
! The BIND statement specifies the BIND attribute for a list of variables and
! common blocks.

module m

  interface
    subroutine proc() bind(c)
    end
  end interface
  procedure(proc), bind(c) :: pc1
  !ERROR: Only variable and named common block can be in BIND statement
  bind(c) :: proc
  !ERROR: Only variable and named common block can be in BIND statement
  bind(c) :: pc1

  !ERROR: Only variable and named common block can be in BIND statement
  bind(c) :: sub

  !PORTABILITY: Name 'm' declared in a module should not have the same name as the module
  bind(c) :: m ! no error for implicit type variable

  type my_type
    integer :: i
  end type
  !ERROR: Only variable and named common block can be in BIND statement
  bind(c) :: my_type

  enum, bind(c) ! no error
    enumerator :: SUNDAY, MONDAY
  end enum

  integer :: x, y, z = 1
  common /blk/ y
  bind(c) :: x, /blk/, z ! no error for variable and common block

  bind(c) :: implicit_i ! no error for implicit type variable

  !ERROR: 'implicit_blk' appears as a COMMON block in a BIND statement but not in a COMMON statement
  bind(c) :: /implicit_blk/

contains

  subroutine sub() bind(c)
  end

end
