! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! sourced allocation with characters of an unlimited polymorphic value

class(*), allocatable :: aa(:)
character*10 :: result
call set(aa)
select type(aa)
type is (character(*))
  result = aa(1) // aa(2)
end select
if (result .ne. " PASS") result = " FAIL"
print*, result
contains
  subroutine set(vv)
    class(*), allocatable, intent(out) :: vv(:)
    allocate(vv, source=[' PAS','S   '])
  end subroutine
end
