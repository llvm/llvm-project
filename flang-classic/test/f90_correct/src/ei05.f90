! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   Initialization expressions containing PARAMETER references

module dd
 type dt
  real :: m
 end type
 type(dt), parameter, dimension(2:6) :: pp = &
   (/ dt(1.), dt(2.5), dt(3), dt(4), dt(5) /)
 type(dt), parameter, dimension(2) :: qq = (/ pp(3:4) /)
end module

use dd
real,dimension(2) :: result, expect
result = qq(:)%m
expect = pp(3:4)%m
call check(result,expect,2)
end
