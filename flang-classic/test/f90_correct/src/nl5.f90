! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! namelist read of a non-initial pointer component

  type tt
    integer :: jj
  end type tt

  type(tt), target  :: aa, bb
  type(tt), pointer :: pp, qq

  namelist /nnn/ pp, qq

  pp => aa
  qq => bb

  aa%jj =  5
  bb%jj = 11

  open(14, file='nl5.dat')
  read(14, nnn)
  close(14)

  if (qq%jj .ne. 17) print*, 'FAIL'
  if (qq%jj .eq. 17) print*, 'PASS'
end
