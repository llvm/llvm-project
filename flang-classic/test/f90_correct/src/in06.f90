!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program test_f90_intrinsics
integer, parameter :: n = 4
integer, parameter :: m = 3
integer, parameter :: l = 5
!
integer, parameter :: ntests = 44
logical, dimension(ntests) :: checks
integer ic
!
integer, dimension(n,m,l) :: ia
logical, dimension(n,m,l) :: la
real,    dimension(n,m,l) :: ra
!
! Answers
logical lan1(m,l), lan2(n,l), lan3(n,m)
logical lal1(m,l), lal2(n,l), lal3(n,m)
integer ico1(m,l), ico2(n,l), ico3(n,m)
integer ima1(m,l), ima2(n,l), ima3(n,m)
integer imi1(m,l), imi2(n,l), imi3(n,m)
real    rma1(m,l), rma2(n,l), rma3(n,m)
real    rmi1(m,l), rmi2(n,l), rmi3(n,m)
integer ipr1(m,l), ipr2(n,l), ipr3(n,m)
real    rpr1(m,l), rpr2(n,l), rpr3(n,m)
integer iad1(m,l), iad2(n,l), iad3(n,m)
real    rad1(m,l), rad2(n,l), rad3(n,m)
!
! Checks
logical clan1(m,l), clan2(n,l), clan3(n,m)
logical clal1(m,l), clal2(n,l), clal3(n,m)
integer cico1(m,l), cico2(n,l), cico3(n,m), cic
integer cima1(m,l), cima2(n,l), cima3(n,m), cma
integer cimi1(m,l), cimi2(n,l), cimi3(n,m), cmi
integer cipr1(m,l), cipr2(n,l), cipr3(n,m), cpr
integer ciad1(m,l), ciad2(n,l), ciad3(n,m), cad

integer iexp(ntests)
!
! Input data
data ia /  1,  4,  7, 10,  2,  5,  8, 11,  3,  6,  9, 12, &
           0,  3,  6,  9,  1,  4,  7, 10,  2,  5,  8, 11, &
          -1,  2,  5,  8,  0,  3,  6,  9,  1,  4,  7, 10, &
          -2,  1,  4,  7, -1,  2,  5,  8,  0,  3,  6,  9, &
          -3,  0,  3,  6, -2,  1,  4,  7, -1,  2,  5,  8 /
!
! Result check data
data clan1 / T, T, T, T, T, T, T, T, T, T, T, T, F, T, T /
data clan2 / F, F, T, T, F, F, T, T, F, F, T, T, F, F, F, T, F, F, F, T /
data clan3 / F, F, T, T, F, F, T, T, F, F, T, T /
!
data clal1 / F, F, F, F, F, F, F, F, F, F, F, F, F, F, F /
data clal2 / F, F, T, T, F, F, F, T, F, F, F, T, F, F, F, T, F, F, F, F /
data clal3 / F, F, F, F, F, F, F, T, F, F, F, T /
!
data cic / 20 /
data cico1 / 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 0, 1, 1 /
data cico2 / 0, 0, 3, 3, 0, 0, 2, 3, 0, 0, 1, 3, 0, 0, 0, 3, 0, 0, 0, 2 /
data cico3 / 0, 0, 1, 4, 0, 0, 2, 5, 0, 0, 3, 5 /
!
data cma / 12 /
data cima1 /10,11,12, 9,10,11, 8, 9,10, 7, 8, 9, 6, 7, 8 /
data cima2 / 3, 6, 9,12, 2, 5, 8,11, 1, 4, 7,10, 0, 3, 6, 9,-1, 2, 5, 8 /
data cima3 / 1, 4, 7,10, 2, 5, 8,11, 3, 6, 9,12 /
!
data cmi / -3 /
data cimi1 / 1, 2, 3, 0, 1, 2,-1, 0, 1,-2,-1, 0,-3,-2,-1 /
data cimi2 / 1, 4, 7,10, 0, 3, 6, 9,-1, 2, 5, 8,-2, 1, 4, 7,-3, 0, 3, 6 /
data cimi3 /-3, 0, 3, 6,-2, 1, 4, 7,-1, 2, 5, 8 /
!
data cpr / 0 /
data cipr1 / 280, 880, 1944, 0, 280, 880, -80, 0, 280, &
             -56, -80, 0, 0, -56, -80 /
data cipr2 / 6, 120, 504, 1320, 0, 60, 336, 990, 0, 24, 210, 720, &
             0, 6, 120, 504, -6, 0, 60, 336 /
data cipr3 / 0, 0, 2520, 30240, 0, 120, 6720, 55440, 0, 720, 15120, 95040 /
!
data cad / 270 /
data ciad1 / 22, 26, 30, 18, 22, 26, 14, 18, 22, 10, 14, 18, 6, 10, 14 /
data ciad2 / 6, 15, 24, 33, 3, 12, 21, 30, 0, 9, 18, 27, &
             -3, 6, 15, 24, -6, 3, 12, 21 /
data ciad3 / -5, 10, 25, 40, 0, 15, 30, 45, 5, 20, 35, 50 /
!
!
!
! Create real and logical arrays
ra = real(ia)
la = ra .gt. 6.0
ic = 1
!
! Do the tests
lan1 = any(la,dim=1)
lan2 = any(la,dim=2)
lan3 = any(la,dim=3)
checks(ic) = .not. any(la); ic = ic + 1
checks(ic) = any(clan1 .neqv. lan1); ic = ic + 1
checks(ic) = any(clan2 .neqv. lan2); ic = ic + 1
checks(ic) = any(clan3 .neqv. lan3); ic = ic + 1
!
lal1 = all(la,dim=1)
lal2 = all(la,dim=2)
lal3 = all(la,dim=3)
checks(ic) = all(la); ic = ic + 1
checks(ic) = any(clal1 .neqv. lal1); ic = ic + 1
checks(ic) = any(clal2 .neqv. lal2); ic = ic + 1
checks(ic) = any(clal3 .neqv. lal3); ic = ic + 1
!
ico1 = count(la,dim=1)
ico2 = count(la,dim=2)
ico3 = count(la,dim=3)
checks(ic) = ((cic - count(la)) .ne. 0); ic = ic + 1
checks(ic) = any((cico1 - ico1) .ne. 0); ic = ic + 1
checks(ic) = any((cico2 - ico2) .ne. 0); ic = ic + 1
checks(ic) = any((cico3 - ico3) .ne. 0); ic = ic + 1
!
ima1 = maxval(ia,dim=1)
ima2 = maxval(ia,dim=2)
ima3 = maxval(ia,dim=3)
checks(ic) = ((cma - maxval(ia)).ne. 0); ic = ic + 1
checks(ic) = any((cima1 - ima1) .ne. 0); ic = ic + 1
checks(ic) = any((cima2 - ima2) .ne. 0); ic = ic + 1
checks(ic) = any((cima3 - ima3) .ne. 0); ic = ic + 1
!
imi1 = minval(ia,dim=1)
imi2 = minval(ia,dim=2)
imi3 = minval(ia,dim=3)
checks(ic) = ((cmi - minval(ia)).ne. 0); ic = ic + 1
checks(ic) = any((cimi1 - imi1) .ne. 0); ic = ic + 1
checks(ic) = any((cimi2 - imi2) .ne. 0); ic = ic + 1
checks(ic) = any((cimi3 - imi3) .ne. 0); ic = ic + 1
!
rma1 = maxval(ra,dim=1)
rma2 = maxval(ra,dim=2)
rma3 = maxval(ra,dim=3)
checks(ic) = ((real(cma) - maxval(ra)).ne. 0); ic = ic + 1
checks(ic) = any((real(cima1) - rma1) .ne. 0); ic = ic + 1
checks(ic) = any((real(cima2) - rma2) .ne. 0); ic = ic + 1
checks(ic) = any((real(cima3) - rma3) .ne. 0); ic = ic + 1
!
rmi1 = minval(ra,dim=1)
rmi2 = minval(ra,dim=2)
rmi3 = minval(ra,dim=3)
checks(ic) = ((real(cmi) - minval(ra)).ne. 0); ic = ic + 1
checks(ic) = any((real(cimi1) - rmi1) .ne. 0); ic = ic + 1
checks(ic) = any((real(cimi2) - rmi2) .ne. 0); ic = ic + 1
checks(ic) = any((real(cimi3) - rmi3) .ne. 0); ic = ic + 1
!
ipr1 = product(ia,dim=1)
ipr2 = product(ia,dim=2)
ipr3 = product(ia,dim=3)
checks(ic) = ((cpr - product(ia)).ne. 0); ic = ic + 1
checks(ic) = any((cipr1 - ipr1) .ne. 0); ic = ic + 1
checks(ic) = any((cipr2 - ipr2) .ne. 0); ic = ic + 1
checks(ic) = any((cipr3 - ipr3) .ne. 0); ic = ic + 1
!
rpr1 = product(ra,dim=1)
rpr2 = product(ra,dim=2)
rpr3 = product(ra,dim=3)
checks(ic) = ((real(cpr) - product(ra)).ne. 0); ic = ic + 1
checks(ic) = any((real(cipr1) - rpr1) .ne. 0); ic = ic + 1
checks(ic) = any((real(cipr2) - rpr2) .ne. 0); ic = ic + 1
checks(ic) = any((real(cipr3) - rpr3) .ne. 0); ic = ic + 1
!
iad1 = sum(ia,dim=1)
iad2 = sum(ia,dim=2)
iad3 = sum(ia,dim=3)
checks(ic) = ((cad - sum(ia)).ne. 0); ic = ic + 1
checks(ic) = any((ciad1 - iad1) .ne. 0); ic = ic + 1
checks(ic) = any((ciad2 - iad2) .ne. 0); ic = ic + 1
checks(ic) = any((ciad3 - iad3) .ne. 0); ic = ic + 1
!
rad1 = sum(ra,dim=1)
rad2 = sum(ra,dim=2)
rad3 = sum(ra,dim=3)
checks(ic) = ((real(cad) - sum(ra)) .ne. 0.0); ic = ic + 1
checks(ic) = any((real(ciad1) - rad1) .ne. 0.0); ic = ic + 1
checks(ic) = any((real(ciad2) - rad2) .ne. 0.0); ic = ic + 1
checks(ic) = any((real(ciad3) - rad3) .ne. 0.0); ic = ic + 1
!
if ((ic-1) .ne. ntests) print *,"Error in number of tests!"
!
do i = 1, ntests
   iexp(i) = 0
end do

call check(checks,iexp,ntests)

!
end
