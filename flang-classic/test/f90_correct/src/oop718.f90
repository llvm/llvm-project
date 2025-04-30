! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test unlimited polymorphic pointer assignment where the target is an unlimited
! polymorphic dummy argument and the corresponding actual argument is character.
! The length must be copied when the pointer is assigned.
module oop718
  implicit none
  class(*), pointer :: val => null()
contains
  subroutine set_val(in_val)
    class(*), target :: in_val
    val => in_val
  end subroutine
end module

use oop718
implicit none

character(len=8), target :: string_val

string_val = 'ABC'
call set_val(string_val)
select type(val)
  type is(character(len=*))
    if (val /= string_val) stop 'FAIL: wrong value'
  class default
    stop 'FAIL: wrong type'
end select
stop 'PASS'

end
