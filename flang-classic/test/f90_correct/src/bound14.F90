!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for LBOUND and UBOUND in data initialization when DIM is present.

program test
  use iso_fortran_env
  implicit none

  integer, parameter :: sp1 = real_kinds(lbound(real_kinds,dim=1))
  integer, parameter :: dp1 = real_kinds(lbound(real_kinds,dim=1)+1)
#ifdef __flang_quadfp__
  integer, parameter :: qp1 = real_kinds(lbound(real_kinds,dim=1)+2)
#endif
  integer, parameter :: sp2 = real_kinds(ubound(real_kinds,dim=1)-2)
  integer, parameter :: dp2 = real_kinds(ubound(real_kinds,dim=1)-1)
#ifdef __flang_quadfp__
  integer, parameter :: qp2 = real_kinds(ubound(real_kinds,dim=1))
#endif

  if (sp1 /= 4 .or. sp2 /= 4) STOP 1
  if (dp1 /= 8 .or. dp2 /= 8) STOP 2
#ifdef __flang_quadfp__
  if (qp1 /= 16 .or. qp2 /= 16) STOP 3
#endif
  print *, "PASS"
end program
