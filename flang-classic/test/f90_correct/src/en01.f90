! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

subroutine foo()
  print *, 'PASS'
end subroutine foo

subroutine bar(msg, func)
  character(len=1) msg
  external func
  return
  entry alt_bar(func)
  call func()
  return
end subroutine bar

interface
   subroutine foo()
   end subroutine foo
end interface

call alt_bar(foo)
end program

