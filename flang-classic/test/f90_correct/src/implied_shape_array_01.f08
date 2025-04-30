! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for implied-shape array

program test
  implicit none
  integer, parameter :: n = 26
  real, parameter :: a1(5:*) = sin((/1.1, 2.2, 3.3, 4.4, 5.5/)) * 1.234
  real, dimension(*), parameter :: a2 = a1
  character(len=*), parameter :: a3(*) = (/character(len=5) ::                 &
                                           "a", "bb", "ccc", "dddd", "eeeee"/)
  integer, parameter :: a4(5:*, *) = reshape((/1, 10, 100, 1000, 10000,        &
                                               2, 20, 200, 2000, 20000/),      &
                                             (/2, 5/))
  integer, parameter :: a5(*, *) = reshape((/1, 10, 100, 1000, 10000,          &
                                             2, 20, 200, 2000, 20000/),        &
                                           (/2, 5/))
  integer, parameter :: a6(*, *, *) = reshape((/1, 2, 3, 4, 5, 6, 7, 8/),      &
                                              (/2, 2, 2/))
  integer :: rslt(n), expt(n)

  expt = (/                    &
           5, 9, 5,            &
           1, 5, 5,            &
           1, 5, 5,            &
           5, 1, 6, 5, 10,     &
           1, 1, 2, 5, 10,     &
           1, 1, 1, 2, 2, 2, 8 &
          /)

  rslt(1:9) = (/                                  &
                lbound(a1), ubound(a1), size(a1), &
                lbound(a2), ubound(a2), size(a2), &
                lbound(a3), ubound(a3), size(a3)  &
               /)
  rslt(10:14) = (/ lbound(a4, dim=1), lbound(a4, dim=2), &
                   ubound(a4, dim=1), ubound(a4, dim=2), size(a4) /)
  rslt(15:19) = (/ lbound(a5, dim=1), lbound(a5, dim=1), &
                   ubound(a5, dim=1), ubound(a5, dim=2), size(a5) /)
  rslt(20:26) = (/ lbound(a6, dim=1), lbound(a6, dim=2), &
                   lbound(a6, dim=3), ubound(a6, dim=1), &
                   ubound(a6, dim=2), ubound(a6, dim=3), size(a6) /)

  call check(rslt, expt, n)

end program test
