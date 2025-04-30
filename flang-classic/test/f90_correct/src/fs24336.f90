! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program nested_types

  use nested_types_module
  implicit none

  type(type2) :: obj
  integer :: int1 = 1
  real :: ret(1)

  obj%member%func1=>used_function
  ret = obj%member%func1(obj%member,int1)
  if (ret(1) .eq. 2.0) then
    print *, "PASS"
  else
    print *, "FAIL"
  endif

end program nested_types
