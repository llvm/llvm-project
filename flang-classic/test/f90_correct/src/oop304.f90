! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod_gen
implicit none
private
type, public :: v
!private
real, allocatable :: r
end type
end module

program p
USE CHECK_MOD
use mod_gen
class(v),allocatable ::stuff
real, allocatable :: r
class(v),allocatable :: stuff2
logical rslt(4)
logical expect(4)

allocate(stuff)
allocate(r)
r = 1.0
expect = .true.
rslt = .false.
allocate(stuff%r,source=r)
!print *, allocated(stuff%r)
rslt(1) = allocated(stuff%r)
allocate(stuff2,source=stuff)
deallocate(stuff%r)
!print *,allocated(stuff2%r),allocated(stuff%r)
rslt(2) = allocated(stuff2%r)
rslt(3) = .not. allocated(stuff%r)
!print *, stuff2
rslt(4) = stuff2%r .eq. 1.0

call check(rslt,expect,4)

end
