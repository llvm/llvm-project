!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for all ENTRY points with the same return type identify the same variable

function f1 ()
  integer :: f1, e1
entry e1 ()
  e1 = 1
end function

function f2 ()
  complex :: f2, e2
entry e2 ()
  e2 = (2, 2)
end function

function f3 ()
  integer, pointer :: f3, e3
entry e3 ()
  allocate(e3)
  e3 = 3
end function

function f4 ()
  integer, dimension(4) :: f4, e4
entry e4 ()
  e4 = (/1,2,3,4/)
end function

function f5 ()
  integer, dimension(:), pointer :: f5, e5
entry e5 ()
  allocate(e5(5))
  e5 = (/1,2,3,4,5/)
end function

function f6 (n)
  integer :: n
  character(n) :: f6, e6
entry e6 (n)
  e6 = "hello1"
end function

function f7 (n)
  integer :: n
  character(n), pointer :: f7, e7
entry e7 (n)
  allocate(e7)
  e7 = "hello2"
end function

program test
  interface
    function f1 ()
      integer :: f1
    end function
    function e1 ()
      integer :: e1
    end function
    function f2 ()
      complex :: f2
    end function
    function e2 ()
      complex :: e2
    end function
    function f3 ()
      integer, pointer :: f3
    end function
    function e3 ()
      integer, pointer :: e3
    end function
    function f4 ()
      integer, dimension(4) :: f4
    end function
    function e4 ()
      integer, dimension(4) :: e4
    end function
    function f5 ()
      integer, dimension(:), pointer :: f5
    end function
    function e5 ()
      integer, dimension(:), pointer :: e5
    end function
    function f6 (n)
      integer :: n
      character(n) :: f6
    end function
    function e6 (n)
      integer :: n
      character(n) :: e6
    end function
    function f7 (n)
      integer :: n
      character(n), pointer :: f7
    end function
    function e7 (n)
      integer :: n
      character(n), pointer :: e7
    end function
  end interface

  integer, parameter :: n = 14
  integer :: rslts(n), expect(n)
  character(6), pointer :: p

  rslts = 0
  expect = 1
  if (f1() == 1) rslts(1) = 1
  if (e1() == 1) rslts(2) = 1
  if (f2() == (2, 2)) rslts(3) = 1
  if (e2() == (2, 2)) rslts(4) = 1
  if (f3() == 3) rslts(5) = 1
  if (e3() == 3) rslts(6) = 1
  if (all(f4() == (/1,2,3,4/))) rslts(7) = 1
  if (all(e4() == (/1,2,3,4/))) rslts(8) = 1
  if (all(f5() == (/1,2,3,4,5/))) rslts(9) = 1
  if (all(e5() == (/1,2,3,4,5/))) rslts(10) = 1
  if (len(f6(6)) == 6 .and. f6(6) == "hello1") rslts(11) = 1
  if (len(e6(6)) == 6 .and. e6(6) == "hello1") rslts(12) = 1
  p => f7(6)
  if (associated(p)) then
    if (len(p) == 6 .and. p == "hello2") rslts(13) = 1
    deallocate(p)
  endif
  p => e7(6)
  if (associated(p)) then
    if (len(p) == 6 .and. p == "hello2") rslts(14) = 1
    deallocate(p)
  endif

  call check(rslts, expect, n)
end program
