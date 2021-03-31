! RUN: %S/test_errors.sh %s %t %f18
! Ensure that references to functions that return pointers can serve as
! "variables" in actual arguments.  All of these uses are conforming and
! no errors should be reported.
module m
  integer, target :: x = 1
 contains
  function get() result(p)
    integer, pointer :: p
    p => x
  end function get
  subroutine increment(n)
    integer, intent(inout) :: n
    n = n + 1
  end subroutine increment
end module m

use m
integer, pointer :: q
get() = 2
call increment(get())
q => get()
read(*) get()
open(file='file',newunit=get())
allocate(q,stat=get())
end
