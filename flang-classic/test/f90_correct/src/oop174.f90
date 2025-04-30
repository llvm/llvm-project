! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

integer function foo(x)
  integer x
  integer, save :: i
  if (x .eq. 0) then
     i = 0
  else
     i = i + x
  endif
  foo = i
end function foo

program p
USE CHECK_MOD
  interface
     integer function foo(x)
       integer x
     end function foo
  end interface
  logical expect(4)
  logical rslt(4)
  integer i
  integer f
  expect = .true.
  rslt = .false.
  i = foo(0)
  associate ( f => foo(1), j=>i)
  rslt(1) = (f .eq. 1)
  rslt(2) = (f .eq. 1)
  rslt(3) = (j .eq. 0)
  i = i + 1
  rslt(4) = (j .eq. 1)
end associate

  call check(rslt,expect,4)

end program p	
