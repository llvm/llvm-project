!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!           c_sizeof() can be applied to elements of a pointer, allocatable, 
!           assumed-shape, and assumed-size array
module bi04
   use iso_c_binding
   integer result(4)
   integer expect(4)
   data expect/4,8,4,8/
   contains
   subroutine sub(aa, pp, yy, zz)
     real, allocatable :: aa(:)
     real, pointer :: pp(:)
     real(8)       :: yy(:)
     real(8)       :: zz(*)
     result(1) = c_sizeof(pp(1))
     result(2) = c_sizeof(yy(1))
     result(3) = c_sizeof(aa(1))
     result(4) = c_sizeof(zz(1))
!     print 99, result
!     print 99, expect
!  99 format(6i4)
     call check(result, expect, 4)
    endsubroutine
end
use bi04
real, allocatable :: aa(:)
real, pointer :: pp(:)
real(8)       :: yy(7)
allocate(aa(0:4))
allocate(pp(3:9))
call sub(aa, pp, yy, yy)
end
