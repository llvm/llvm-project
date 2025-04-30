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

subroutine baseproc(this,this2,rslt)
class(base_t) :: this
class(base_t), optional :: this2
logical, intent(out) :: rslt

if (present(this2)) then 
    rslt = .true.
else 
    rslt = .false.
endif 
end subroutine

end module

program p
USE CHECK_MOD
use mod
logical results(2)
logical expect(2)
data results /.true.,.false./
data expect /.false.,.true./
type(base_t) :: t

call t%some_proc(rslt=results(1))
call t%some_proc(t,rslt=results(2))

call check(results,expect,2)

end program 
