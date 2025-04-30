!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!


subroutine s1 ! error
  implicit none
  integer :: x(10)
  real :: results
  integer :: i

  do i=1,10
    x(i) = i
  enddo
  !{error "PGF90-S-0074-Illegal number or type of arguments to norm2 - keyword argument x"}
  results = norm2(x)
end

subroutine s2 ! error
  implicit none
  real :: y(10)
  real :: results
  integer :: i
  do i=1,10
    y(i)=i
  end do
  !{error "PGF90-S-0423-Constant DIM= argument is out of range"}
  results = norm2(y, 2)
end

subroutine s3
  implicit none
  real :: z(10)
  real :: results
  integer :: i
  do i=1,10
    z(i)=i
  end do
  !{error "PGF90-S-0074-Illegal number or type of arguments to norm2 - keyword argument position 3"}
  results = norm2(z, 1, 0)
end

subroutine s4
  implicit none
  real :: z(10)
  real :: results
  integer :: i
  do i=1,10
    z(i)=i
  end do
  !{error "PGF90-S-0074-Illegal number or type of arguments to norm2 - 0 argument(s) present, 1-3 argument(s) expected"}
  results = norm2()
end

subroutine s5
  implicit none
  !{error "PGF90-S-0155-Intrinsic not supported in initialization: norm2"}
  real :: results = norm2([real :: 2, 3, 4, 5])
end

  call s1
  call s2
  call s3
  call s4
  call s4
  call s5
end
