!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   REAL with complex arguments and with/without KIND argument

program accuracy
  implicit none
  complex(kind=8)   :: z8
  complex(kind=4)   :: z4
  real(kind=8)      :: result(8), expect(8)
  data expect/ 0.5555555555555556d0,0.5555555555555556d0,0.5555555555555556d0, &
	       0.5555555820465088d0,0.5555555820465088d0,0.5555555820465088d0, &
	       0.5555555820465088d0,0.5555555820465088d0/

  
  z8 = (0.5555555555555555555555555d0,0.55555555555555555555555d0)
  result(1) = real(z8)
  result(2) = dreal(z8)
  result(3) = real(z8,kind=8)
  result(4) = real(z8,kind=4)
  z4 = (0.5555555555555555555555555e0,0.55555555555555555555555e0)
  result(5) = real(z4)
  result(6) = dreal(z4)
  result(7) = real(z4,kind=8)
  result(8) = real(z4,kind=4)
!  print *,result
!  print *,expect
  call check(result,expect,16)
end program accuracy
