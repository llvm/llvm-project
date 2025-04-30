! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! final routine compilation

module mm
  type :: tt
  contains
    final :: ff
  end type tt
contains
  subroutine ff(dd)
    type(tt) :: dd
    print*, 'PASS'
  end
  subroutine ss
    type(tt) :: xx
  end
end

  use mm
  call ss
end
