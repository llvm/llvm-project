! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
type base_t
contains
procedure :: some_proc => baseproc
end type

contains

logical function baseproc(this,this2)
class(base_t) :: this
class(base_t), optional :: this2

if (present(this2)) then 
    baseproc = .true.
else 
    baseproc = .false.
endif 
end function

end module

program p
USE CHECK_MOD
use mod
logical results(2)
logical expect(2)
data results /.true.,.false./
data expect /.false.,.true./
type(base_t) :: t

results(1) = t%some_proc()
results(2) = t%some_proc(t)

call check(results,expect,2)

end program 
