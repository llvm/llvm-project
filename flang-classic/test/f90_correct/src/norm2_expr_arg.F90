!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program norm2_expr_arg

  parameter(N=1)
  integer :: result(N),expect(N)
  real    :: x(10), y(10)
  integer :: i

  do i = 1, 10
    x(i) = i * i
    y(i) = i
  enddo

  result(1) = norm2(x-y)
  expect(1) = 140.2426
  call check(result,expect,N)

end program
