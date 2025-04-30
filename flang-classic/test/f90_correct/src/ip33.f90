! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!   Internal subprogram.
!   The main program contains only the call to the internal subprogram,
!   and there is no program name (i.e., 'program p').

  call f()

contains
  subroutine f()
    parameter(n=1)
    integer result(n), expect(n)
    data expect/2/
    b = 2
    result(1) = b
    call check( result, expect, n )
  end subroutine
end

