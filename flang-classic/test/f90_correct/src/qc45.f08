! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for the usage of c_long_double_complex.
! Complex(c_long_double_complex) is equal to complex(16).

program example
  use, intrinsic :: iso_c_binding

  complex(c_long_double_complex) :: res
  complex(16) :: a, b, c

  a = (44.6441301697361925144580643835703598_16, &
       7.03794069978169795284658606936286925_16)
  b = (0.0_16, 0.0_16)
  res = (44.6441301697361925144580643835703598_16, &
         7.03794069978169795284658606936286925_16)
  c = a - res

  if ((c == b) .neqv. .true.) stop 1
  if (C_SIZEOF(res) /= 32) stop 2

  print *, 'PASS'
end
