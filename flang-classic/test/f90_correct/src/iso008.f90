! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


subroutine foo(arg)
  use iso_c_binding
  integer, pointer :: z
  type(c_ptr), value, intent(in)  :: arg
  call c_f_pointer(arg, z);
  if (z .eq. -99) then
    print *, 'PASS'
  else
    print *, 'FAIL'
  endif
end subroutine foo

program p
  use iso_c_binding
  
  interface
     subroutine foo(arg)
       use iso_c_binding
       type(c_ptr), value, intent(in)  :: arg
     end subroutine foo

     subroutine func(arg) bind(c)
       use iso_c_binding
       type(c_funptr), value, intent(in)  :: arg
     end subroutine func
  end interface
  
  abstract interface
     subroutine ifoo(arg)
       use iso_c_binding
       type(c_ptr), value, intent(in)  :: arg
     end subroutine ifoo
  end interface
  
  
  procedure(ifoo), bind(c), pointer :: proc
  type(c_funptr) :: tmp_cptr
  
  proc => foo
  
  tmp_cptr = c_funloc(proc)
  
  call func(tmp_cptr)
end program p
