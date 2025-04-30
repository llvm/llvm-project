! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! check that passing intrinsics works with or without interface blocks

program tp0023

  real(kind = 16) a, b, c, d, e
  logical :: rslt(6) = [1,1,1,1,1,1]
  logical :: expect(6) = [1,1,1,1,1,1]
  a = 1.1_16
  b = 1.2_16
  c = subs1(b, a)

  if (c /= 2.3_16) then
    rslt(1) = 0
  endif

  d = subs2(b, a)
  if (c /= d) then
    rslt(2) = 0
  endif

  call subs3(a, b, e)
  if (e /= d) then
    rslt(3) = 0
  endif

  c = subs1(1.1_16, 1.2_16)
  if (c /= 2.3_16) then
    rslt(4) = 0
  endif

  d = subs2(1.1_16,b)
  if (c /= d) then
    rslt(5) = 0
  endif

  call subs3(a, 1.2_16, e)
  if (e /= d) then
    rslt(6) = 0
  endif

  call check(rslt,expect,6)

contains

  function subs1(a, b ) result(c)
    real(kind = 16), intent(in) :: a, b
    real(kind = 16) :: c
    c = a + b
  end function

  real(16) function  subs2(a, b)
    real(kind = 16), intent(in) :: a, b
    subs2 = a + b
  end function

  subroutine subs3(a, b, c)
    real(kind = 16), intent(in)  :: a, b
    real(kind = 16), intent(out) :: c
    c = a + b
  end subroutine

end program
