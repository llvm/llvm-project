! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module test_module
use, intrinsic :: iso_c_binding, only: &
  c_ptr, &
  c_loc
implicit none
private :: c_ptr
private :: c_loc

contains 
subroutine test_c_loc(value,cptr)
  integer, target :: value
  type(c_ptr) :: cptr
  cptr = c_loc(value)
end subroutine

end module
