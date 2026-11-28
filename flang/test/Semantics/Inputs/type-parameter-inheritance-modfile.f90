module m1
  type base
    integer, private :: c
  contains
    private
    procedure, nopass, public :: f
    generic, private :: g => f
    generic :: h => f
    generic, public :: public_g => f
  end type
contains
  subroutine f
  end subroutine
end module

module m2
  use m1
  type, extends(base) :: t(c)
    integer, len :: c
  end type
  type, extends(base) :: private_generics(g, h)
    integer, len :: g, h
  end type
end module
