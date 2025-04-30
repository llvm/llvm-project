! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module nested_types_module

  implicit none
!  private
  
  public type1, type2 

  type type1 
     procedure(used_function), pointer, nopass :: func1 => null()
  end type type1

  type type2 
     type(type1) :: member
  end type type2
  contains

     function unused_function() result(ret_val)
       integer :: ret_val
       ret_val = 2
     end function unused_function

     function used_function(thing1, int1) result(ret_val)
       type(type1), intent(inout) :: thing1
       integer, intent(in) :: int1
       real :: ret_val(int1) 
       ret_val = 2.0
     end function used_function

    
end module nested_types_module
