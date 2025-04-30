! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module z
USE CHECK_MOD
    contains
        subroutine print_me(mm)
        type rarr
        complex, pointer :: cmmm(:,:,:,:)
        endtype
	integer results(1)
	integer expect(1)
        type (rarr), allocatable :: data_p(:)
	type (rarr) :: p
	results = .false.
	expect = .true.
        allocate(data_p(mm))
	results(1) = same_type_as(p,data_p)
	call check(results,expect,1)
	
        endsubroutine
    end
    use z
    call print_me(10)
    end 

