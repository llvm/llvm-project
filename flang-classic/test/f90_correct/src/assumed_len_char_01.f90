!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for function which return assumed-length character.

character(len=*) function func1()
  func1 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
end function func1

character(len=*) function func2(str3)
  character(len=3), intent(in) :: str3
  func2 = str3//'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
end function func2

subroutine test1(i)
  integer :: i
  character(len=i) :: str
  character(len=i) :: func1
  str = func1()
  if (str .ne. 'ABCDEFGHIJKLMNOPQRST') STOP 1
end subroutine test1

subroutine test2(i)
  integer :: i
  character(len=i) :: str
  character(len=i), external :: func1
  str = func1()
  if (str .ne. 'ABCDEFGHIJKLMNOPQRST') STOP 2
end subroutine test2

subroutine test3(i, str4)
  integer :: i
  character(len=3) :: str4
  character(len=i) :: str
  character(len=i) :: func2
  str = func2(str4)
  if (str .ne. '$*$ABCDEFGHIJKLMNOPQ') STOP 3
end subroutine test3

subroutine test4(i, str5)
  integer :: i
  character(len=3) :: str5
  character(len=i) :: str
  character(len=i), external :: func2
  str = func2(str5)
  if (str .ne. '## ABCDEFGHIJKLMNOPQ') STOP 4
end subroutine test4

program p
  character(len=3) :: str1, str2
  str1 = '$*$'
  str2 = '##'
  call test1(20)
  call test2(20)
  call test3(20, str1)
  call test4(20, str2)
  print *, 'PASS'
end program
