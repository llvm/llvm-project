! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! array constructor with allocated element values

module mm
  type t_aa
    integer, allocatable :: ii(:)
  end type t_aa

contains
  integer function ff(aa)
    type(t_aa) :: aa(:)

    ff = 0
    do i = lbound(aa,1), ubound(aa,1)
      do j = lbound(aa(i)%ii,1), ubound(aa(i)%ii,1)
        ff = ff + aa(i)%ii(j)
      enddo
    enddo
  end function ff

  subroutine rr(kk)
    integer            :: kk, nn
    type(t_aa), target :: aa

    allocate(aa%ii(1))
    aa%ii = 7
    nn = ff((/(aa, j=1, kk)/))

    if (aa%ii(1) .eq. 7 .and. nn .eq. 35) then
      print*, 'PASS'
    else
      print*, 'FAIL'
    endif
  end subroutine rr
end module mm

  use mm
  call rr(5)
end
