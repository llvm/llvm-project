! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Check that passing intrinsics works with or without interface blocks.

program tp0023
  use check_mod
  intrinsic sin, cos
  real(kind = 16)  pi, res(4), exp(4)
  data exp/0.0_16, 1.0_16, 0.0_16, 1.0_16/
  interface
    subroutine  SUBR(RF1, ARG, RES )
      real(kind = 16)  ARG, RES
      interface
        function  RF1(RX)
          real(kind = 16)  RF1
          real(kind = 16)  RX
        end function
      end interface
    end subroutine
  end interface

  pi = acos(1.0)
  call subs(sin, pi, res(1))
  call subs(cos, pi, res(2))
  call subs(sin, pi, res(3))
  call subs(cos, pi, res(4))
  !write(*,*)res(1:4)
  call checkr16(res,exp,4)
end

subroutine SUBR( RF1, ARG, RES )
  real(kind=16)  ARG, RES
  interface
    function  RF1(RX)
      real(kind=16)  RF1
      real(kind=16)  RX
    end function
  end interface
  RES = RF1(ARG)
end subroutine

subroutine SUBS( RF1, ARG, RES )
  real(kind=16)  ARG, RES
  external RF1
  RES = RF1(ARG)
end subroutine
