! Code based on the reproducer from https://github.com/flang-compiler/flang/issues/1014

! Declare module a which contains subroutine sub
module a
  implicit none
  interface sub
    module procedure sub
  end interface

contains

  subroutine sub(i)
    implicit none
    integer, intent(in) :: i

    ! Make sure that the sub subroutine is linked and called correctly.
    integer :: result(1)
    integer :: expect(1)
    result(1) = i
    expect(1) = 10
    call check(result, expect, 1)
  end subroutine

end module

! module b uses a, so now sub is available through a and b
module b
  use a
  implicit none
  private
  public :: sub
end module

! module c uses b - so now sub is available though a, b and c
module c
  use b
  implicit none
end module

program x
  use c, sub2=> sub ! rename sub to sub2...
  use b             ! ... but then use b - sub is still available
  implicit none

  call sub(10) ! call sub (thanks to "use b")
end program
