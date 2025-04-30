! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test of arbitrary constant double word to complex(16)

program test
  integer, parameter :: n = 4
  integer, parameter :: m = n * 3
  complex(16) :: result(m), expect(m)
  complex(16), parameter :: rst1 = z'1234567812'
  complex(16), parameter :: rst2 = z'123456781234'
  complex(16), parameter :: rst3 = z'12345678123456'
  complex(16), parameter :: rst4 = z'1234567812345678'
  complex(16) :: rst5 = z'1234567812'
  complex(16) :: rst6 = z'123456781234'
  complex(16) :: rst7 = z'12345678123456'
  complex(16) :: rst8 = z'1234567812345678'

  expect(1) = (4.240075845258268272609759187713423E-4954_16, 0.0_16)
  expect(2) = (1.085459416386880748452192039017725E-4951_16, 0.0_16)
  expect(3) = (2.778776105950420155184711947826470E-4949_16, 0.0_16)
  expect(4) = (7.113666831233075629648738183625889E-4947_16, 0.0_16)
  expect(5) = (5.062777118760537493528106882795804E-4955_16, 0.0_16)
  expect(6) = (1.296070942406064689405303135053406E-4952_16, 0.0_16)
  expect(7) = (3.317941612559581291383603192752674E-4950_16, 0.0_16)
  expect(8) = (8.493930528152528882963038506009860E-4948_16, 0.0_16)
  expect(9) = (5.062777118760537493528106882795804E-4955_16, 0.0_16)
  expect(10) = (1.296070942406064689405303135053406E-4952_16, 0.0_16)
  expect(11) = (3.317941612559581291383603192752674E-4950_16, 0.0_16)
  expect(12) = (8.493930528152528882963038506009860E-4948_16, 0.0_16)

  result(1) = z'9876543298'
  result(2) = z'987654329876'
  result(3) = z'98765432987654'
  result(4) = z'9876543298765432'
  result(n+1) = rst1
  result(n+2) = rst2
  result(n+3) = rst3
  result(n+4) = rst4
  result(n+5) = rst5
  result(n+6) = rst6
  result(n+7) = rst7
  result(n+8) = rst8

  do i = 1, m
    if (result(i) .ne. expect(i)) STOP i
  enddo

  print *, 'PASS'

end program
