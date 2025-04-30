! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   Initialization expressions containing PARAMETER references

module cc
 real, parameter, dimension(5) :: pp = (/ 1,2,3,4,5 /)
 real, parameter, dimension(10) :: qq = (/ pp(1:5),pp(1:5) /)
end module

use cc
real,expect(10)
expect(1:5) = pp
expect(6:10) = pp
call check(qq,expect,10)
end
