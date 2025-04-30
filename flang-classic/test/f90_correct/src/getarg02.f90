!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for semantic checks about getarg

subroutine check_argcnt()
  integer(kind=4) :: pos
  character(len=32) :: arg, extra

  pos = 1

  !{error "PGF90-S-0074-Illegal number or type of arguments to getarg - 3 argument(s) present, 2 argument(s) expected"}
  call getarg(pos, arg, extra)

end subroutine

subroutine check_arg1()
  real(kind=4) :: pos
  character(len=32) :: arg

  pos = 1.0
  !{error "PGF90-S-0074-Illegal number or type of arguments to getarg - keyword argument pos"}
  call getarg(pos, arg)

end subroutine

subroutine check_arg2()
  integer(kind=4) :: pos
  character(len=10), parameter :: arg = '123456'

  pos = 1

  !{error "PGF90-S-0074-Illegal number or type of arguments to getarg - keyword argument *value"}
  call getarg(pos, arg)

end subroutine

subroutine check_arg3()
  integer(kind=4) :: pos, arg

  pos = 1

  !{error "PGF90-S-0074-Illegal number or type of arguments to getarg - keyword argument *value"}
  call getarg(pos, arg)

end subroutine
