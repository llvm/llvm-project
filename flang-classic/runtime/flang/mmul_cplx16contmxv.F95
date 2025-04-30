! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"

subroutine F90_matmul_cplx16_contmxv(dest, src1, src2, k_extent, m_extent)

  DESC_INT k_extent
  DESC_INT m_extent
  COMPLEX*16, dimension(k_extent,m_extent) :: src1
  COMPLEX*16, dimension(m_extent) :: src2
  COMPLEX*16, dimension(k_extent) :: dest

  DESC_INT k
  DESC_INT m

  do k=1,k_extent
    dest(k) = 0
  end do
  do m=1,m_extent
    do k=1,k_extent
      dest(k) = dest(k) + src1(k,m) * src2(m)
    end do
  end do

end subroutine
