! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!
!  Simple target associated test.
!  Watch out for -Mchkptr
module m
   type t
      complex, pointer, dimension(:) :: p => null()
      complex, allocatable, dimension(:)  :: q
   end type t
   contains
     subroutine init(x)
       type(t), intent(inout) :: x
       nullify(x%p)
     end subroutine init
     subroutine check_assoc(x)
       type(t), intent(inout), target :: x
       if( associated(x%p,x%q) ) then
	  call check(0,1,1)
          write(*,*) "Yes"
       else
          write(*,*) "No"
	  call check(1,1,1)
       end if
     end subroutine check_assoc
   end module m
program assoc_test
   use m
   type(t) :: x
   call init(x)
   call check_assoc(x)
end program assoc_test
