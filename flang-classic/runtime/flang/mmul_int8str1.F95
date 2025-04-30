! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 



#include "mmul_dir.h"

subroutine F90_matmul_int8_str1(dest,s1,s2, &
      k_extnt,m_extnt,n_extnt,                   &
      s1_d1_extnt,s2_d1_extnt,d_d1_extnt,        &
      d_d1_lstride)

      DESC_INT  n_extnt,m_extnt,k_extnt
      DESC_INT  s1_d1_extnt,s2_d1_extnt,d_d1_extnt,d_d1_lstride
      INTEGER   bs
      parameter (bs=192)
      INTEGER*8 s1(s1_d1_extnt,m_extnt)
      INTEGER*8 s2(s2_d1_extnt,k_extnt)
      INTEGER*8 dest(d_d1_extnt,n_extnt*d_d1_lstride)
      DESC_INT  i, j, l, nmod4, mmod4
      DESC_INT  ii,jj,ll,temppos, nn, kk, itest
      INTEGER*8   t00,t01,t02,t03
      INTEGER*8   t10,t11,t12,t13
      INTEGER*8   t20,t21,t22,t23
      INTEGER*8   t30,t31,t32,t33
      INTEGER*8 temp0, temp1, temp2, temp3, temp (bs*bs)
      INTEGER*8 s20, s21, s22, s23
      INTEGER   flag
      INTEGER*8 zero
      parameter (zero = 0)

      DESC_INT k,n,m

      if (d_d1_lstride .eq. 1) then
         nmod4 = mod (k_extnt, 4)
         mmod4 = mod (n_extnt, 4)
         kk = k_extnt - nmod4
         nn = n_extnt - mmod4
         itest = nmod4 + mmod4
         do jj=1,k_extnt,bs
          do ii=1,n_extnt,bs
           flag = 0
           do ll=1,m_extnt,bs
            temppos = 1
            if ((k_extnt .ge. 4) .and. (n_extnt .ge. 4)) then
            if (jj .le. kk) then
             j = jj
             do i=ii,min(nn,ii+bs-1),4
              if (flag .eq. 0) then
               t00=zero
               t01=zero
               t02=zero
               t03=zero
               t10=zero
               t11=zero
               t12=zero
               t13=zero
               t20=zero
               t21=zero
               t22=zero
               t23=zero
               t30=zero
               t31=zero
               t32=zero
               t33=zero
              else
               t00=dest(i+0,j+0)
               t01=dest(i+0,j+1)
               t02=dest(i+0,j+2)
               t03=dest(i+0,j+3)
               t10=dest(i+1,j+0)
               t11=dest(i+1,j+1)
               t12=dest(i+1,j+2)
               t13=dest(i+1,j+3)
               t20=dest(i+2,j+0)
               t21=dest(i+2,j+1)
               t22=dest(i+2,j+2)
               t23=dest(i+2,j+3)
               t30=dest(i+3,j+0)
               t31=dest(i+3,j+1)
               t32=dest(i+3,j+2)
               t33=dest(i+3,j+3)
              end if
              do l=ll,min(m_extnt,ll+bs-1)
               temp0 = s1(i+0,l)
               temp1 = s1(i+1,l)
               temp2 = s1(i+2,l)
               temp3 = s1(i+3,l)
               s20 = s2(l,j+0)
               s21 = s2(l,j+1)
               s22 = s2(l,j+2)
               s23 = s2(l,j+3)
               t00=t00+s20*temp0
               t01=t01+s21*temp0
               t02=t02+s22*temp0
               t03=t03+s23*temp0
               t10=t10+s20*temp1
               t11=t11+s21*temp1
               t12=t12+s22*temp1
               t13=t13+s23*temp1
               t20=t20+s20*temp2
               t21=t21+s21*temp2
               t22=t22+s22*temp2
               t23=t23+s23*temp2
               t30=t30+s20*temp3
               t31=t31+s21*temp3
               t32=t32+s22*temp3
               t33=t33+s23*temp3
               temp (temppos+0) = temp0
               temp (temppos+1) = temp1
               temp (temppos+2) = temp2
               temp (temppos+3) = temp3
               temppos = temppos + 4
              end do 
              dest(i+0,j+0)=t00
              dest(i+0,j+1)=t01
              dest(i+0,j+2)=t02
              dest(i+0,j+3)=t03
              dest(i+1,j+0)=t10
              dest(i+1,j+1)=t11
              dest(i+1,j+2)=t12
              dest(i+1,j+3)=t13
              dest(i+2,j+0)=t20
              dest(i+2,j+1)=t21
              dest(i+2,j+2)=t22
              dest(i+2,j+3)=t23
              dest(i+3,j+0)=t30
              dest(i+3,j+1)=t31
              dest(i+3,j+2)=t32
              dest(i+3,j+3)=t33
             end do 
            do j=jj+4,min(kk,jj+bs-1),4
             temppos = 1
             do i=ii,min(nn,ii+bs-1),4
              if (flag .eq. 0) then
               t00=zero
               t01=zero
               t02=zero
               t03=zero
               t10=zero
               t11=zero
               t12=zero
               t13=zero
               t20=zero
               t21=zero
               t22=zero
               t23=zero
               t30=zero
               t31=zero
               t32=zero
               t33=zero
              else
               t00=dest(i+0,j+0)
               t01=dest(i+0,j+1)
               t02=dest(i+0,j+2)
               t03=dest(i+0,j+3)
               t10=dest(i+1,j+0)
               t11=dest(i+1,j+1)
               t12=dest(i+1,j+2)
               t13=dest(i+1,j+3)
               t20=dest(i+2,j+0)
               t21=dest(i+2,j+1)
               t22=dest(i+2,j+2)
               t23=dest(i+2,j+3)
               t30=dest(i+3,j+0)
               t31=dest(i+3,j+1)
               t32=dest(i+3,j+2)
               t33=dest(i+3,j+3)
              endif
              do l=ll,min(m_extnt,ll+bs-1)
               temp0 = temp (temppos+0)
               temp1 = temp (temppos+1)
               temp2 = temp (temppos+2)
               temp3 = temp (temppos+3)
               s20 = s2(l,j+0)
               s21 = s2(l,j+1)
               s22 = s2(l,j+2)
               s23 = s2(l,j+3)
               t00=t00+s20*temp0
               t01=t01+s21*temp0
               t02=t02+s22*temp0
               t03=t03+s23*temp0
               t10=t10+s20*temp1
               t11=t11+s21*temp1
               t12=t12+s22*temp1
               t13=t13+s23*temp1
               t20=t20+s20*temp2
               t21=t21+s21*temp2
               t22=t22+s22*temp2
               t23=t23+s23*temp2
               t30=t30+s20*temp3
               t31=t31+s21*temp3
               t32=t32+s22*temp3
               t33=t33+s23*temp3
               temppos = temppos + 4
              end do 
              dest(i+0,j+0)=t00
              dest(i+0,j+1)=t01
              dest(i+0,j+2)=t02
              dest(i+0,j+3)=t03
              dest(i+1,j+0)=t10
              dest(i+1,j+1)=t11
              dest(i+1,j+2)=t12
              dest(i+1,j+3)=t13
              dest(i+2,j+0)=t20
              dest(i+2,j+1)=t21
              dest(i+2,j+2)=t22
              dest(i+2,j+3)=t23
              dest(i+3,j+0)=t30
              dest(i+3,j+1)=t31
              dest(i+3,j+2)=t32
              dest(i+3,j+3)=t33
             end do 
            end do 
            end if
            end if
            if (itest .ne. 0) then
            if (nmod4 .ne. 0) then
             temppos = 1
             do j=kk+1,min(kk+1,jj+bs-1)
              do i=ii,min(nn,ii+bs-1),4
               if (flag .eq. 0) then
                t00=zero
                t10=zero
                t20=zero
                t30=zero
               else
                t00=dest(i+0,j+0)
                t10=dest(i+1,j+0)
                t20=dest(i+2,j+0)
                t30=dest(i+3,j+0)
               end if
               do l=ll,min(m_extnt,ll+bs-1)
                temp0 = s1(i+0,l)
                temp1 = s1(i+1,l)
                temp2 = s1(i+2,l)
                temp3 = s1(i+3,l)
                t00=t00+s2(l,j+0)*temp0
                t10=t10+s2(l,j+0)*temp1
                t20=t20+s2(l,j+0)*temp2
                t30=t30+s2(l,j+0)*temp3
                temp (temppos+0) = temp0
                temp (temppos+1) = temp1
                temp (temppos+2) = temp2
                temp (temppos+3) = temp3
                temppos = temppos + 4
               end do 
               dest(i+0,j+0)=t00
               dest(i+1,j+0)=t10
               dest(i+2,j+0)=t20
               dest(i+3,j+0)=t30
              end do 
             end do
             do j=kk+2,min(k_extnt,jj+bs-1)
              temppos = 1
              do i=ii,min(nn,ii+bs-1),4
               if (flag .eq. 0) then
                t00=zero
                t10=zero
                t20=zero
                t30=zero
               else
                t00=dest(i+0,j+0)
                t10=dest(i+1,j+0)
                t20=dest(i+2,j+0)
                t30=dest(i+3,j+0)
               end if
               do l=ll,min(m_extnt,ll+bs-1)
                temp0 = temp (temppos+0)
                temp1 = temp (temppos+1)
                temp2 = temp (temppos+2)
                temp3 = temp (temppos+3)
                t00=t00+s2(l,j+0)*temp0
                t10=t10+s2(l,j+0)*temp1
                t20=t20+s2(l,j+0)*temp2
                t30=t30+s2(l,j+0)*temp3
                temppos = temppos + 4
               end do 
               dest(i+0,j+0)=t00
               dest(i+1,j+0)=t10
               dest(i+2,j+0)=t20
               dest(i+3,j+0)=t30
              end do 
             end do 
            end if
            if (mmod4 .ne. 0) then
             temppos = 1
             if (jj .le. kk) then
             j = jj
             do i=nn+1,min(n_extnt,ii+bs-1)
              if (flag .eq. 0) then
               t00=zero
               t01=zero
               t02=zero
               t03=zero
              else
               t00=dest(i+0,j+0)
               t01=dest(i+0,j+1)
               t02=dest(i+0,j+2)
               t03=dest(i+0,j+3)
              end if
              do l=ll,min(m_extnt,ll+bs-1)
               temp0 = s1(i+0,l)
               t00=t00+s2(l,j+0)*temp0
               t01=t01+s2(l,j+1)*temp0
               t02=t02+s2(l,j+2)*temp0
               t03=t03+s2(l,j+3)*temp0
               temp (temppos) = temp0
               temppos = temppos + 1
              end do 
              dest(i+0,j+0)=t00
              dest(i+0,j+1)=t01
              dest(i+0,j+2)=t02
              dest(i+0,j+3)=t03
             end do 
             do j=jj+4,min(kk,jj+bs-1),4
              temppos = 1
              do i=nn+1,min(n_extnt,ii+bs-1)
               if (flag .eq. 0) then
                t00=zero
                t01=zero
                t02=zero
                t03=zero
               else
                t00=dest(i+0,j+0)
                t01=dest(i+0,j+1)
                t02=dest(i+0,j+2)
                t03=dest(i+0,j+3)
               end if
               do l=ll,min(m_extnt,ll+bs-1)
                temp0 = temp (temppos)
                t00=t00+s2(l,j+0)*temp0
                t01=t01+s2(l,j+1)*temp0
                t02=t02+s2(l,j+2)*temp0
                t03=t03+s2(l,j+3)*temp0
                temppos = temppos + 1
               end do 
               dest(i+0,j+0)=t00
               dest(i+0,j+1)=t01
               dest(i+0,j+2)=t02
               dest(i+0,j+3)=t03
              end do 
             end do 
             end if
            end if
            if ((nmod4 .ne. 0) .and. (mmod4 .ne. 0)) then
             temppos = 1
             do j=kk+1,min(kk+1,jj+bs-1)
              do i=nn+1,min(n_extnt,ii+bs-1)
               if (flag .eq. 0) then
                t00=zero
               else
                t00=dest(i,j)
               end if
               do l=ll,min(m_extnt,ll+bs-1)
                temp0 = s1(i,l)
                t00=t00+s2(l,j)*temp0
                temp (temppos) = temp0
                temppos = temppos + 1
               end do 
               dest(i,j)=t00
              end do 
             end do 
             do j=kk+2,min(k_extnt,jj+bs-1)
              temppos = 1
              do i=nn+1,min(n_extnt,ii+bs-1)
               if (flag .eq. 0) then
                t00=zero
               else
                t00=dest(i,j)
               end if
               do l=ll,min(m_extnt,ll+bs-1)
                temp0 = temp (temppos)
                t00=t00+s2(l,j)*temp0
                temppos = temppos + 1
               end do 
               dest(i,j)=t00
              end do 
             end do 
            end if
            end if
            flag = 1
           end do 
          end do 
         end do 
      else
         do k = 1, k_extnt
            do n = 1, n_extnt
               dest(1+(n-1)*d_d1_lstride,k) = 0.0d0
            enddo
         enddo
         do k = 1, k_extnt
           do m = 1, m_extnt
               do n = 1, n_extnt
                  dest(1+(n-1)*d_d1_lstride,k) =                &
                                 dest(1+(n-1)*d_d1_lstride,k) + &
                                            s1(n,m) * s2(m,k)
               enddo
            enddo
         enddo
      endif
      return
      end

subroutine F90_matmul_int8_str1_mxv(dest, s1,s2,  &
                    n_extent,m_extent, ld1,dlstride)

   implicit none
   DESC_INT  n_extent,m_extent,ld1,ld2,dlstride
   DESC_INT  mmod4, mmod2, m2
   DESC_INT  jx,kx,jj,kk,kmod4,k4,incx
   DESC_INT  j0,j1,j2,j3,iy,ky,m4
   INTEGER*8 t0,t1,t2,t3
   INTEGER*8 s1(ld1,m_extent)
   INTEGER*8 s2(m_extent)
   INTEGER*8 dest(ld1)

   DESC_INT  i,j,k
   INTEGER   bs
   parameter (bs = 384)
   INTEGER*8 temp (bs)
   DESC_INT  ind(bs)
   INTEGER*8 zero
   parameter (zero = 0)

   if (dlstride .eq. 1) then
         do k = 1, n_extent
            dest(k) = 0.0d0
         end do
         mmod4 = mod(n_extent, 4)
         mmod2 = mod(n_extent, 2)
         m4 = n_extent - mmod4
         m2 = n_extent- mmod2
         kx = 1
         ky = 1
         incx = 1
         jx = kx
         do jj = 1, m_extent, bs
            jx = kx + (jj-1)
            kk = 0
            do j = jj, min (m_extent, jj+bs-1)
               if (s2(jx) .ne. zero) then
                  kk = kk + 1
                  temp(kk) = s2(jx)
                  ind(kk) = j
               end if
               jx = jx + 1
            end do
            kmod4 = mod(kk, 4)
            k4 = kk - kmod4
            do j = 1, k4, 4
               t0 = temp(j)
               t1 = temp(j+1)
               t2 = temp(j+2)
               t3 = temp(j+3)
               j0 = ind(j)
               j1 = ind(j+1)
               j2 = ind(j+2)
               j3 = ind(j+3)
               iy = ky
               do i = 1, m4, 4
                  dest( iy ) = dest( iy )+t0*s1(i, j0) &
                           + t1*s1(i, j1) &
                           + t2*s1(i, j2) &
                           + t3*s1(i, j3)

                  dest(iy+1) = dest(iy+1)+t0*s1(i+1, j0) &
                               + t1*s1(i+1, j1) &
                               + t2*s1(i+1, j2) &
                               + t3*s1(i+1, j3)

                  dest(iy+2) = dest(iy+2)+t0*s1(i+2, j0) &
                               + t1*s1(i+2, j1) &
                               + t2*s1(i+2, j2) &
                               + t3*s1(i+2, j3)

                  dest(iy+3) = dest(iy+3 )+t0*s1(i+3, j0) &
                               + t1*s1(i+3, j1) &
                               + t2*s1(i+3, j2) &
                               + t3*s1(i+3, j3)
                  iy = iy + 4
               end do
               iy = ky + m4
               do i = m4 + 1, n_extent
                  dest(iy) = dest(iy) + t0*s1(i, j0) &
                           + t1*s1(i, j1) &
                           + t2*s1(i, j2) &
                           + t3*s1(i, j3)
                  iy = iy + 1
               end do
            end do
            do j = k4+1, kk
               t0 = temp(j)
               j0 = ind(j)
               iy = ky
               do i = 1, m4, 4
                  dest( iy ) = dest(iy) +t0*s1(i,j0)
                  dest( iy+1) = dest(iy+1) +t0*s1(i+1,j0)
                  dest( iy+2) = dest(iy+2)+t0*s1(i+2,j0)
                  dest( iy+3) = dest(iy+3)+t0*s1(i+3,j0)
                  iy = iy + 4
              end do
           end do
           do j = k4+1, kk
              t0 = temp(j)
              j0 = ind(j)
              iy = ky + m4
              do i = m4 + 1, n_extent
                dest(iy) = dest(iy) + t0 * s1(i, j0)
                iy = iy + 1
              end do
           end do

         end do
   else
         do k = 1, n_extent
            dest(1+(k-1)*dlstride) = 0.0d0
         enddo
         mmod4 = mod(n_extent, 4)
         mmod2 = mod(n_extent, 2)
         m4 = n_extent - mmod4
         m2 = n_extent- mmod2
         kx = 1
         ky = 1
         incx = 1
         jx = kx
         do jj = 1, m_extent, bs
            jx = kx + (jj-1)
            kk = 0
            do j = jj, min (m_extent, jj+bs-1)
               if (s2(jx) .ne. zero) then
                  kk = kk + 1
                  temp(kk) = s2(jx)
                  ind(kk) = j
               end if
               jx = jx + 1
            end do
            kmod4 = mod(kk, 4)
            k4 = kk - kmod4
            do j = 1, k4, 4
               t0 = temp(j)
               t1 = temp(j+1)
               t2 = temp(j+2)
               t3 = temp(j+3)
               j0 = ind(j)
               j1 = ind(j+1)
               j2 = ind(j+2)
               j3 = ind(j+3)
               iy = ky
               do i = 1, m4, 4
                  dest(1+(iy-1)*dlstride) = dest(1+(iy-1)*dlstride) + t0*s1(i, j0) &
                           + t1*s1(i, j1) &
                           + t2*s1(i, j2) &
                           + t3*s1(i, j3)
   
                  dest(1+(iy + 1 -1)*dlstride) = dest(1+(iy + 1 -1)*dlstride) + t0*s1(i+1, j0) &
                               + t1*s1(i+1, j1) &
                               + t2*s1(i+1, j2) &
                               + t3*s1(i+1, j3)
   
                  dest(1+(iy + 2 -1)*dlstride) = dest(1+(iy + 2 -1)*dlstride) + t0*s1(i+2, j0) &
                               + t1*s1(i+2, j1) &
                               + t2*s1(i+2, j2) &
                               + t3*s1(i+2, j3)
   
                  dest(1+(iy + 3 -1)*dlstride) = dest(1+(iy + 3 -1)*dlstride) + t0*s1(i+3, j0) &
                               + t1*s1(i+3, j1) &
                               + t2*s1(i+3, j2) &
                               + t3*s1(i+3, j3)
                  iy = iy + 4
               end do
               iy = ky + m4
               do i = m4 + 1, n_extent
                  dest(1+(iy-1)*dlstride) = dest(1+(iy-1)*dlstride) + t0*s1(i, j0) &
                           + t1*s1(i, j1) &
                           + t2*s1(i, j2) &
                           + t3*s1(i, j3)
                  iy = iy + 1
               end do
            end do
            do j = k4+1, kk
               t0 = temp(j)
               j0 = ind(j)
               iy = ky
               do i = 1, m4, 4
                  dest(1+(iy-1)*dlstride) = dest(1+(iy-1)*dlstride) + t0*s1(i,j0)
                  dest(1+(iy + 1 -1)*dlstride) = dest(1+(iy + 1 -1)*dlstride) + t0*s1(i+1,j0)
                  dest(1+(iy + 2 -1)*dlstride) = dest(1+(iy + 2 -1)*dlstride) + t0*s1(i+2,j0)
                  dest(1+(iy + 3 -1)*dlstride) = dest(1+(iy + 3 -1)*dlstride) + t0*s1(i+3,j0)
                  iy = iy + 4
              end do
           end do
           do j = k4+1, kk
              t0 = temp(j)
              j0 = ind(j)
              iy = ky + m4
              do i = m4 + 1, n_extent
                dest(1+(iy-1)*dlstride) = dest(1+(iy-1)*dlstride) + t0 * s1(i, j0)
                iy = iy + 1
              end do
           end do
         end do
   endif
end

subroutine F90_matmul_int8_str1_vxm(dest, s1,s2,  &
                   k_extent,m_extent, ld1,dlstride)

   implicit none
   DESC_INT  k_extent,m_extent,ld1,ld2,dlstride
   INTEGER*8 s1(m_extent)
   INTEGER*8 s2(ld1,k_extent)
   INTEGER*8 dest(k_extent)

   INTEGER   bs
   parameter (bs = 384)
   INTEGER*8 temp (bs)
   INTEGER*8 t0,t1,t2,t3
   INTEGER*8 t4,t5,t6,t7
   INTEGER*8 dtemp0, dtemp1, dtemp2, dtemp3
   INTEGER*8 dtemp4, dtemp5, dtemp6, dtemp7
   DESC_INT  ind(bs),ind0,ind1,ind2,ind3
   DESC_INT  ind4,ind5,ind6,ind7
   INTEGER*8 zero
   parameter (zero = 0)

   DESC_INT  mi,ki,ti
   DESC_INT  ii,jj,j,jx,kk,jnext
   DESC_INT  mmod8,m8,kmod8,k8,tmod8,tt8
   DESC_INT  mmod4,m4,kmod4,k4,tmod4,tt4

   if (dlstride .eq. 1) then
      do ki = 1, k_extent
         dest(ki) = 0
       end do

      mmod8 = mod(m_extent,8)
      m8 = m_extent - mmod8
      kmod8 = mod(k_extent,8)
      k8 = k_extent - kmod8

      do kk = 1,k8,8
         dtemp0 = dest(kk)
         dtemp1 = dest(kk+1)
         dtemp2 = dest(kk+2)
         dtemp3 = dest(kk+3)
         dtemp4 = dest(kk+4)
         dtemp5 = dest(kk+5)
         dtemp6 = dest(kk+6)
         dtemp7 = dest(kk+7)
         jnext = 1
         do jj = 1,m8,bs
            ! load s1 temp vector
            ti = 0
            jx = jj
            do j = jj, min (m_extent, jj+bs-1)
               if (s1(jx) .ne. zero) then
                  ti = ti + 1
                  temp(ti) = s1(jx)
                  ind(ti) = j
               end if
               jx = jx + 1
            end do

            tmod8 = mod(ti,8)
            tt8 = ti - tmod8

            if (tt8 .ne. 0) then
               jnext = ind(tt8)+1
            endif

            ti = 1
            do ii = 1,tt8,8
               t0 =  temp(ti)
               t1 =  temp(ti+1)
               t2 =  temp(ti+2)
               t3 =  temp(ti+3)
               t4 =  temp(ti+4)
               t5 =  temp(ti+5)
               t6 =  temp(ti+6)
               t7 =  temp(ti+7)
               ind0 = ind(ti)
               ind1 = ind(ti+1)
               ind2 = ind(ti+2)
               ind3 = ind(ti+3)
               ind4 = ind(ti+4)
               ind5 = ind(ti+5)
               ind6 = ind(ti+6)
               ind7 = ind(ti+7)
   
               dtemp0 = dtemp0 + &
                          t0 * s2(ind0, kk) + t1 * s2(ind1, kk) + &
                          t2 * s2(ind2, kk) + t3 * s2(ind3, kk) + &
                          t4 * s2(ind4, kk) + t5 * s2(ind5, kk) + &
                          t6 * s2(ind6, kk) + t7 * s2(ind7, kk) 
               dtemp1 = dtemp1 + &
                          t0 * s2(ind0, kk+1) + t1 * s2(ind1, kk+1) + &
                          t2 * s2(ind2, kk+1) + t3 * s2(ind3, kk+1) + &
                          t4 * s2(ind4, kk+1) + t5 * s2(ind5, kk+1) + &
                          t6 * s2(ind6, kk+1) + t7 * s2(ind7, kk+1) 
               dtemp2 = dtemp2 +  &
                          t0 * s2(ind0, kk+2) + t1 * s2(ind1, kk+2) + &
                          t2 * s2(ind2, kk+2) + t3 * s2(ind3, kk+2) + &
                          t4 * s2(ind4, kk+2) + t5 * s2(ind5, kk+2) + &
                          t6 * s2(ind6, kk+2) + t7 * s2(ind7, kk+2) 
               dtemp3 = dtemp3 +  &
                          t0 * s2(ind0, kk+3) + t1 * s2(ind1, kk+3) + &
                          t2 * s2(ind2, kk+3) + t3 * s2(ind3, kk+3) + &
                          t4 * s2(ind4, kk+3) + t5 * s2(ind5, kk+3) + &
                          t6 * s2(ind6, kk+3) + t7 * s2(ind7, kk+3) 
   
               dtemp4 = dtemp4 + &
                          t0 * s2(ind0, kk+4) + t1 * s2(ind1, kk+4) + &
                          t2 * s2(ind2, kk+4) + t3 * s2(ind3, kk+4) + &
                          t4 * s2(ind4, kk+4) + t5 * s2(ind5, kk+4) + &
                          t6 * s2(ind6, kk+4) + t7 * s2(ind7, kk+4) 
               dtemp5 = dtemp5 + &
                          t0 * s2(ind0, kk+5) + t1 * s2(ind1, kk+5) + &
                          t2 * s2(ind2, kk+5) + t3 * s2(ind3, kk+5) + &
                          t4 * s2(ind4, kk+5) + t5 * s2(ind5, kk+5) + &
                          t6 * s2(ind6, kk+5) + t7 * s2(ind7, kk+5) 
               dtemp6 = dtemp6 +  &
                          t0 * s2(ind0, kk+6) + t1 * s2(ind1, kk+6) + &
                          t2 * s2(ind2, kk+6) + t3 * s2(ind3, kk+6) + &
                          t4 * s2(ind4, kk+6) + t5 * s2(ind5, kk+6) + &
                          t6 * s2(ind6, kk+6) + t7 * s2(ind7, kk+6) 
               dtemp7 = dtemp7 +  &
                          t0 * s2(ind0, kk+7) + t1 * s2(ind1, kk+7) + &
                          t2 * s2(ind2, kk+7) + t3 * s2(ind3, kk+7) + &
                          t4 * s2(ind4, kk+7) + t5 * s2(ind5, kk+7) + &
                          t6 * s2(ind6, kk+7) + t7 * s2(ind7, kk+7) 
               ti = ti + 8
            enddo
         enddo
         do ii = jnext,m_extent
            t0 = s1(ii)
            dtemp0 = dtemp0 + t0 * s2(ii,kk)
            dtemp1 = dtemp1 + t0 * s2(ii,kk+1)
            dtemp2 = dtemp2 + t0 * s2(ii,kk+2)
            dtemp3 = dtemp3 + t0 * s2(ii,kk+3)
            dtemp4 = dtemp4 + t0 * s2(ii,kk+4)
            dtemp5 = dtemp5 + t0 * s2(ii,kk+5)
            dtemp6 = dtemp6 + t0 * s2(ii,kk+6)
            dtemp7 = dtemp7 + t0 * s2(ii,kk+7)
         enddo
         dest(kk) = dtemp0
         dest(kk+1) = dtemp1
         dest(kk+2) = dtemp2
         dest(kk+3) = dtemp3
         dest(kk+4) = dtemp4
         dest(kk+5) = dtemp5
         dest(kk+6) = dtemp6
         dest(kk+7) = dtemp7
      enddo
      do kk = k8+1,k_extent
         dtemp0 = dest(kk)
         do ii = 1,m_extent
            dtemp0 = dtemp0 + s1(ii) *  s2(ii,kk)
          enddo
         dest(kk) = dtemp0 
      enddo


   else
      do kk = 1, k_extent
         dest(1+(kk-1)*dlstride) = 0
      end do

      mmod4 = mod(m_extent,4)
      m4 = m_extent - mmod4
      kmod4 = mod(k_extent,4)
      k4 = k_extent - kmod4

      do kk = 1,k4,4
         dtemp0 = dest(1+(kk-1)*dlstride)
         dtemp1 = dest(1+(kk)*dlstride)
         dtemp2 = dest(1+(kk+1)*dlstride)
         dtemp3 = dest(1+(kk+2)*dlstride)
         jnext = 1
         do jj = 1,m4,bs
            ! load s1 temp vector
            ti = 0
            jx = jj
            do j = jj, min (m_extent, jj+bs-1)
               if (s1(jx) .ne. zero) then
                  ti = ti + 1
                  temp(ti) = s1(jx)
                  ind(ti) = j
               end if
               jx = jx + 1
            end do

            tmod4 = mod(ti,4)
            tt4 = ti - tmod4

            if (tt4 .ne. 0) then
               jnext = ind(tt4)+1
            endif

            ti = 1
            do ii = 1,tt4,4
               t0 =  temp(ti)
               t1 =  temp(ti+1)
               t2 =  temp(ti+2)
               t3 =  temp(ti+3)
               ind0 = ind(ti)
               ind1 = ind(ti+1)
               ind2 = ind(ti+2)
               ind3 = ind(ti+3)

               dtemp0 = dtemp0 + &
                          t0 * s2(ind0, kk) + t1 * s2(ind1, kk) + &
                          t2 * s2(ind2, kk) + t3 * s2(ind3, kk) 
               dtemp1 = dtemp1 + &
                            t0 * s2(ind0, kk+1) + t1 * s2(ind1, kk+1) + &
                            t2 * s2(ind2, kk+1) + t3 * s2(ind3, kk+1) 
               dtemp2 = dtemp2 +  &
                            t0 * s2(ind0, kk+2) + t1 * s2(ind1, kk+2) + &
                            t2 * s2(ind2, kk+2) + t3 * s2(ind3, kk+2)
               dtemp3 = dtemp3 +  &
                            t0 * s2(ind0, kk+3) + t1 * s2(ind1, kk+3) + &
                            t2 * s2(ind2, kk+3) + t3 * s2(ind3, kk+3) 
   
               ti = ti + 4
            enddo
         enddo
         do ii = jnext,m_extent
            t0 = s1(ii)
            dtemp0 = dtemp0 +  t0 * s2(ii,kk)
            dtemp1 = dtemp1 + t0 * s2(ii,kk+1)
            dtemp2 = dtemp2 + t0 * s2(ii,kk+2)
            dtemp3 = dtemp3 +  t0 * s2(ii,kk+3)
         enddo
         dest(1+(kk-1)*dlstride) = dtemp0
         dest(1+(kk)*dlstride) = dtemp1
         dest(1+(kk+1)*dlstride) = dtemp2
         dest(1+(kk+2)*dlstride) = dtemp3
      enddo
      do kk = k4+1,k_extent
         dtemp0 = dest(1+(kk-1)*dlstride)
         do ii = 1,m_extent
            dtemp0 = dtemp0 + s1(ii) *  s2(ii,kk)
          enddo
         dest(1+(kk-1)*dlstride) = dtemp0 
      enddo
   endif
end

