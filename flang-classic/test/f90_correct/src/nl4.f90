!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! automatic arrays in namelist

SUBROUTINE S1(I, yy, output) 
integer :: yy(i), zz(I), aa(I), ii
NAMELIST /NLIST/ yy, zz, ii
character*40 output(8)
ii    = 5
zz(1) = 3
zz(2) = 4
!write(6     ,NML=NLIST)
write(output,NML=NLIST)
END SUBROUTINE S1

program test
integer aa(3)
data aa/1,2,-99/
character*40 output(8)
integer yy(2), zz(2), ii
NAMELIST /NLIST/ yy, zz, ii
integer result(5)
integer expect(5)
data expect/5,1,2,3,4/
call S1(2,aa,output)
!write(6, '(a)') output
read(output, nlist)
!write(6, '(6i4)') yy, zz, ii

result(1) = ii
result(2:3) = yy
result(4:5) = zz
call check(result, expect, 5)
end
