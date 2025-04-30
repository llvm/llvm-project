! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"

subroutine F90_matmul_log2_contvxm(dest, src1, src2, m_extent, n_extent)

  DESC_INT n_extent
  DESC_INT m_extent
  LOGICAL*2, dimension(m_extent) :: src1
  LOGICAL*2, dimension(m_extent,n_extent) :: src2
  LOGICAL*2, dimension(n_extent) :: dest

  DESC_INT n
  DESC_INT m

  do n=1,n_extent
    dest(n) = 0
    do m=1,m_extent
      if ( src1(m) .AND. src2(m,n) ) then
        dest(n) = .TRUE.
        cycle
      endif
    end do
 end do

end subroutine
