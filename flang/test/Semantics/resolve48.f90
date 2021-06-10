! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test correct use-association of a derived type.
module m1
  implicit none
  type :: t
  end type
end module
module m2
  use m1, only: t
end module
module m3
  use m2
  type(t) :: o
end

! Test access-stmt with generic interface and type of same name.
module m4
  private
  public :: t1, t2
  type :: t2
  end type
  interface t1
    module procedure init1
  end interface
  interface t2
    module procedure init2
  end interface
  type :: t1
  end type
contains
  type(t1) function init1()
  end function
  type(t2) function init2()
  end function
end module
