! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! sourced allocation with characters of an unlimited polymorphic value
! 
! same as oop731 except uses multiple sourced allocations

class(*), allocatable :: aa(:)
class(*), allocatable :: bb(:)
character*10 :: result
call set(aa,bb)
select type(aa)
type is (character(*))
  result = aa(1) // aa(2)
class default
  result = " FAIL"
end select
if (result .ne. " PASS") result = " FAIL"
if (result .eq. " PASS") then
  select type(bb)
  type is (character(*))
    result = bb(1) // bb(2)
  class default
    result = " FAIL"
  end select
endif
if (result .ne. " PASS") result = " FAIL"
print*, result
contains
  subroutine set(vv,uu)
    class(*), allocatable, intent(out) :: vv(:)
    class(*), allocatable, intent(out) :: uu(:)
    allocate(vv,  uu, source=[' PAS','S   '])
  end subroutine
end
