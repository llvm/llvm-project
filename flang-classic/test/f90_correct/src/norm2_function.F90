!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


program test
implicit none
  real, parameter :: tolerance = 1.0E-12
  integer, parameter :: n = 3
  integer, parameter :: expc (2) = [1, 1]
  integer :: res(2)
  real :: x(n)
  real :: resultsA(3)
  real :: expectA(3)
  integer i

  do i=1, n
    x(i) = i*i
  enddo
  ! Test for assumed size
  expectA(1) = sqrt(dot_product(x, x))
  resultsA(1) = norm2ComputeDummy(n, x)

  ! Test for adjustable array
  expectA(2) = sqrt(dot_product(x, x))
  resultsA(2) = norm2AdjustableArray(n, x)

  ! Test for assumed size
  expectA(3) = sqrt(dot_product(x(1:2), x(1:2)))
  resultsA(3) = norm2AssumedSize(x(1:2))

  if(all(abs(resultsA - expectA) < tolerance)) then
    res(1) = 1
    print *, 'expect vs results match'
  else
    res(1) = -1
    print *, 'expect vs results mismatch'
  end if

  res(2) = checkLocal(norm2(x), expectA(1))
  call check(res, expc, 2)
contains

function checkLocal(norm2In, expectIn)
  implicit none
  real :: norm2In
  real :: expectIn
  integer :: checkLocal

  if (abs(norm2In-expectIn)< tolerance) then
    print *, 'expect vs results match'
    checkLocal = 1
  else
    print *, 'expect vs results mismatch'
    checkLocal = -1
  end if
end function

function norm2ComputeDummy(sz, inData)
  implicit none
  integer :: sz
  real, dimension(*) :: inData
  real :: norm2ComputeDummy

  norm2ComputeDummy = norm2(inData(1:sz))
end function


function norm2AdjustableArray(sz, inData)
  implicit none
  integer :: sz
  real, dimension(sz) :: inData
  real :: norm2AdjustableArray

  norm2AdjustableArray = norm2(inData)
end function

function norm2AssumedSize(inData)
  implicit none
  real, dimension(:) :: inData
  real :: norm2AssumedSize

  norm2AssumedSize = norm2(inData)
end function

end program

