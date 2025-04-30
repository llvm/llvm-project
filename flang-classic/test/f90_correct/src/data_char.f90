! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for character variable initialized by DATA statement.

program p
  character*6 a
  character*10 b
  character*10 c
  data a(2:5) /'AAA'/
  data b(3:6) /'BBB'/
  data b(8:9) /'CC'/
  data c(3:5) /'345'/
  data c(6:9) /'6789'/

  if (a .ne. ' AAA  ') stop 1
  if (b .ne. '  BBB  CC ') stop 2
  if (c .ne. '  3456789 ') stop 3

  print *, 'PASS'
end
