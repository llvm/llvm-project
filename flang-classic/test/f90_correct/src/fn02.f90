!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! If there is a RESULT clause in a function header, the result name
! means the result value, the function name means the function itself

module bug

contains

  recursive function foo (i) result (foo_x)
   integer foo_x
   integer i
   foo_x = i
   call bar(foo,foo_x)
  end function foo

  recursive subroutine bar (f,i)
   integer i
   interface
    recursive function f (i) result (f_x)
     integer f_x
     integer i
    end function
   end interface
   if( i .gt. 0 ) then
    j = f(i-1)
    i = i + j
   endif
  end subroutine

end module bug

use bug
integer expect(1),result(1)
result(1) = foo(3)
expect(1) = 6
call check(result,expect,1)
end
