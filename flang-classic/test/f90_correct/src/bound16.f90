!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test the fix for the LBOUND/UBOUND regression, where the LBOUND/UBOUND call
! nested within the SIZE call has a defer-shape ARRAY parameter and a DIM
! parameter.

module m
  implicit none
  type t1
    character, pointer :: arr1(:)
  end type
  type t2
    type(t1), pointer :: arr2(:)
    integer, pointer :: arr3(:)
  end type
contains
  function test_default(a) result(ret)
    type(t2) :: a
    character(len=size(a%arr2(ubound(a%arr3, 1))%arr1)) :: ret
    ret = repeat('a', len(ret))
  end function

  function test_kind1(a) result(ret)
    type(t2) :: a
    character(len=size(a%arr2(ubound(a%arr3, 1, kind=1))%arr1)) :: ret
    ret = repeat('b', len(ret))
  end function

  function test_kind2(a) result(ret)
    type(t2) :: a
    character(len=size(a%arr2(ubound(a%arr3, 1, kind=2))%arr1)) :: ret
    ret = repeat('c', len(ret))
  end function

  function test_kind4(a) result(ret)
    type(t2) :: a
    character(len=size(a%arr2(ubound(a%arr3, 1, kind=4))%arr1)) :: ret
    ret = repeat('d', len(ret))
  end function

  function test_kind8(a) result(ret)
    type(t2) :: a
    character(len=size(a%arr2(ubound(a%arr3, 1, kind=8))%arr1)) :: ret
    ret = repeat('e', len(ret))
  end function
end module

program test
  use m
  implicit none
  type(t2) :: x
  character(len=:), allocatable :: y

  allocate(x%arr2(5))
  allocate(x%arr3(2:4))
  allocate(x%arr2(4)%arr1(2:10))
  y = test_default(x)
  if (len(y) /= 9 .or. y /= 'aaaaaaaaa') STOP 1
  y = test_kind1(x)
  if (len(y) /= 9 .or. y /= 'bbbbbbbbb') STOP 2
  y = test_kind2(x)
  if (len(y) /= 9 .or. y /= 'ccccccccc') STOP 3
  y = test_kind4(x)
  if (len(y) /= 9 .or. y /= 'ddddddddd') STOP 4
  y = test_kind8(x)
  if (len(y) /= 9 .or. y /= 'eeeeeeeee') STOP 5
  print *, "PASS"
end program
