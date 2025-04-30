!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for the import module from the correct path. This file is in the wrong
! path for testing.

module import_source
contains
  subroutine sub1()
  end

  subroutine sub3()
  end

  function func() result(funit)
    integer :: funit
  end
end
