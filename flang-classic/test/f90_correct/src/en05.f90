!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the fix for introduced error, in which the entry returns adjustable
! length character.

function f1(n)
  integer :: n
  character(n) :: f1, e1
  f1 = "AB"
  return
entry e1(n)
  e1 = "XYZ"
end function

function f2(n)
  integer :: n
  character(n), pointer :: f2, e2
  allocate(f2)
  f2 = "AB"
  return
entry e2(n)
  allocate(e2)
  e2 = "XYZ"
end function

function f3(n)
  integer :: n
  character(n), dimension(2, 2) :: f3, e3
  f3 = "AB"
  return
entry e3(n)
  e3 = "XYZ"
end function

program p
  interface
    function f1(n)
      integer :: n
      character(n) :: f1
    end function
    function e1(n)
      integer :: n
      character(n) :: e1
    end function
    function f2(n)
      integer :: n
      character(n), pointer :: f2
    end function
    function e2(n)
      integer :: n
      character(n), pointer :: e2
    end function
    function f3(n)
      integer :: n
      character(n), dimension(2, 2) :: f3
    end function
    function e3(n)
      integer :: n
      character(n), dimension(2, 2) :: e3
    end function
  end interface
  character(:), pointer :: cp

  if (len(f1(2)) /= 2 .or. f1(2) /= "AB") STOP 1
  if (len(e1(3)) /= 3 .or. e1(6) /= "XYZ") STOP 2

  cp => f2(2)
  if (.not. associated(cp)) STOP 3
  if (len(cp) /= 2 .or. cp /= "AB") STOP 4
  deallocate(cp)

  cp => e2(3)
  if (.not. associated(cp)) STOP 5
  if (len(cp) /= 3 .or. cp /= "XYZ") STOP 6
  deallocate(cp)

  if (len(f3(2)) /= 2 .or. any(f3(2) /= "AB")) STOP 7
  if (len(e3(3)) /= 3 .or. any(e3(3) /= "XYZ")) STOP 8

  print *, "PASS"
end program
