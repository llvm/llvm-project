! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"

subroutine F90_matmul_log4_contmxm(dest, src1, src2, k_extent, m_extent, n_extent)

  DESC_INT k_extent
  DESC_INT n_extent
  DESC_INT m_extent
  LOGICAL*4, dimension(k_extent,m_extent) :: src1
  LOGICAL*4, dimension(m_extent,n_extent) :: src2
  LOGICAL*4, dimension(k_extent,n_extent) :: dest

  DESC_INT k
  DESC_INT n
  DESC_INT m

  do n=1,n_extent
    do k=1,k_extent
      dest(k,n) = 0
    end do
  end do
  do n=1,n_extent
    do m=1,m_extent
      do k=1,k_extent
        if ( src1(k,m) .AND. src2(m,n) ) then
          dest(k,n) = .TRUE.
          cycle
        endif
      end do
    end do
  end do

end subroutine
