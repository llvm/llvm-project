! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! typed forall indexes

integer, parameter :: N = 6

integer :: i = -33
integer :: a(N) = 0
integer :: s = 0

forall (integer*1 :: i=N:1:-1) a(i) =  1; s = s + sum(a)/N  !  1
forall (integer*2 :: i=N:1:-1) a(i) =  2; s = s + sum(a)/N  !  3
forall (             i=N:1:-1) a(i) =  3; s = s + sum(a)/N  !  6
forall (integer*8 :: i=N:1:-1) a(i) =  4; s = s + sum(a)/N  ! 10
forall (integer*1 :: i=N:1:-1) a(i) =  5; s = s + sum(a)/N  ! 15
forall (integer*2 :: i=N:1:-1) a(i) =  6; s = s + sum(a)/N  ! 21
forall (             i=N:1:-1) a(i) =  7; s = s + sum(a)/N  ! 28
forall (             i=N:1:-1) a(i) =  8; s = s + sum(a)/N  ! 36
forall (integer*4 :: i=N:1:-1) a(i) =  9; s = s + sum(a)/N  ! 45
forall (integer*8 :: i=N:1:-1) a(i) = 10; s = s + sum(a)/N  ! 55

if (s .ne. 55 .or.  i .ne. -33) print*, 'FAIL'
if (s .eq. 55 .and. i .eq. -33) print*, 'PASS'
end
