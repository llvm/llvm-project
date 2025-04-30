/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* mmulLOG8.c --F90  MATMUL intrinsics for logical*8 type */

#include "stdioInterf.h"
#include "fioMacros.h"

void ENTF90(MATMUL_LOG8, matmul_log8)(char *dest_addr, char *s1_addr,
                                      char *s2_addr, F90_Desc *dest_desc,
                                      F90_Desc *s1_desc, F90_Desc *s2_desc)
{

  __LOG8_T *s1_base;
  __LOG8_T *s2_base;
  __LOG8_T *dest_base;
  __LOG8_T *d_elem_p;
  __LOG8_T *s1_elem_p;
  __LOG8_T *s2_elem_p;

  __INT_T s1_d1_lstride;
  __INT_T s1_d1_sstride;
  __INT_T s1_d1_lb;
  __INT_T s1_d1_soffset = 0;

  __INT_T s1_d2_lstride = 1;
  __INT_T s1_d2_sstride = 1;
  __INT_T s1_d2_lb = 0;
  __INT_T s1_d2_soffset = 0;

  __INT_T s2_d1_lstride;
  __INT_T s2_d1_sstride;
  __INT_T s2_d1_lb;
  __INT_T s2_d1_soffset = 0;

  __INT_T s2_d2_lstride = 1;
  __INT_T s2_d2_sstride = 1;
  __INT_T s2_d2_lb = 0;
  __INT_T s2_d2_soffset = 0;

  __INT_T d_d1_lstride;
  __INT_T d_d1_sstride;
  __INT_T d_d1_lb;
  __INT_T d_d1_soffset = 0;

  __INT_T d_d2_lstride = 1;
  __INT_T d_d2_sstride = 1;
  __INT_T d_d2_lb = 0;
  __INT_T d_d2_soffset = 0;

  __INT_T d_rank = F90_RANK_G(dest_desc);
  __INT_T s1_rank = F90_RANK_G(s1_desc);
  __INT_T s2_rank = F90_RANK_G(s2_desc);

  __INT_T k_extent = s2_rank == 2 ? F90_DIM_EXTENT_G(s2_desc, 1) : 1;
  __INT_T m_extent = s1_rank == 2 ? F90_DIM_EXTENT_G(s1_desc, 1)
                                  : F90_DIM_EXTENT_G(s1_desc, 0);
  __INT_T n_extent = s1_rank == 2 ? F90_DIM_EXTENT_G(s1_desc, 0) : 1;

  __INT_T dest_offset;

  __INT_T s1_d1_base, s1_d1_delta, s1_d1_offset, s1_d2_base, s1_d2_delta,
      s1_d2_offset, s2_d1_base, s2_d1_delta, s2_d1_offset, s2_d2_base,
      s2_d2_delta, s2_d2_offset, d_d1_base, d_d1_delta, d_d1_offset, d_d2_base,
      d_d2_delta, d_d2_offset;

  __INT_T k;
  __INT_T m;
  __INT_T n;

  /* mxm
   *  s1(n,m) x s2(m,k) -> dest(n,k)
   *  Check
   *   dest_d1 extent== n_extnet
   *   dest_d2 extent == k_extent
   *   s2_d1 extent = m_extent
   *
   * mxv
   *  s1(n,m) x s2(m) -> dest(n)
   *  Check
   *   dest_d1 extent== n_extent
   *   s2_d1 extent == m_extent
   *
   * vxm
   *  s1(m) x s2(m,k) -> dest(k)
   *  check
   *   s2_d1 extent == m_extent
   *   dest_d1 extent == k_extent
   */

  if (d_rank == 2 && s1_rank == 2 && s2_rank == 2) {
    if (F90_DIM_EXTENT_G(dest_desc, 0) != n_extent ||
        F90_DIM_EXTENT_G(dest_desc, 1) != k_extent ||
        F90_DIM_EXTENT_G(s2_desc, 0) != m_extent) {
      __fort_abort("MATMUL: nonconforming array shapes");
    }
  } else if (d_rank == 1 && s1_rank == 2 && s2_rank == 1) {
    if (F90_DIM_EXTENT_G(dest_desc, 0) != n_extent ||
        F90_DIM_EXTENT_G(s2_desc, 0) != m_extent) {
      __fort_abort("MATMUL: nonconforming array shapes");
    }
  } else if (d_rank == 1 && s1_rank == 1 && s2_rank == 2) {
    if (F90_DIM_EXTENT_G(dest_desc, 0) != k_extent ||
        F90_DIM_EXTENT_G(s2_desc, 0) != m_extent) {
      __fort_abort("MATMUL: nonconforming array shapes");
    }
  } else {
    __fort_abort("MATMUL: non-conforming array shapes");
  }

  s1_d1_lstride = F90_DIM_LSTRIDE_G(s1_desc, 0);
  s1_d1_sstride = F90_DIM_SSTRIDE_G(s1_desc, 0);
  s1_d1_lb = F90_DIM_LBOUND_G(s1_desc, 0);
  if (s1_d1_sstride != 1 || F90_DIM_SOFFSET_G(s1_desc, 0))
    s1_d1_soffset = F90_DIM_SOFFSET_G(s1_desc, 0) + s1_d1_sstride - s1_d1_lb;

  if (s1_rank == 2) {
    s1_d2_lstride = F90_DIM_LSTRIDE_G(s1_desc, 1);
    s1_d2_lb = F90_DIM_LBOUND_G(s1_desc, 1);
    s1_d2_sstride = F90_DIM_SSTRIDE_G(s1_desc, 1);
    if (s1_d2_sstride != 1 || F90_DIM_SOFFSET_G(s1_desc, 1))
      s1_d2_soffset = F90_DIM_SOFFSET_G(s1_desc, 1) + s1_d2_sstride - s1_d2_lb;
  }

  s2_d1_lstride = F90_DIM_LSTRIDE_G(s2_desc, 0);
  s2_d1_lb = F90_DIM_LBOUND_G(s2_desc, 0);
  s2_d1_sstride = F90_DIM_SSTRIDE_G(s2_desc, 0);
  if (s2_d1_sstride != 1 || F90_DIM_SOFFSET_G(s2_desc, 0))
    s2_d1_soffset = F90_DIM_SOFFSET_G(s2_desc, 0) + s2_d1_sstride - s2_d1_lb;

  if (s2_rank == 2) {
    s2_d2_lstride = F90_DIM_LSTRIDE_G(s2_desc, 1);
    s2_d2_lb = F90_DIM_LBOUND_G(s2_desc, 1);
    s2_d2_sstride = F90_DIM_SSTRIDE_G(s2_desc, 1);
    if (s2_d2_sstride != 1 || F90_DIM_SOFFSET_G(s2_desc, 1))
      s2_d2_soffset = F90_DIM_SOFFSET_G(s2_desc, 1) + s2_d2_sstride - s2_d2_lb;
  }

  d_d1_lstride = F90_DIM_LSTRIDE_G(dest_desc, 0);
  d_d1_lb = F90_DIM_LBOUND_G(dest_desc, 0);
  d_d1_sstride = F90_DIM_SSTRIDE_G(dest_desc, 0);
  if (d_d1_sstride != 1 || F90_DIM_SOFFSET_G(dest_desc, 0))
    d_d1_soffset = F90_DIM_SOFFSET_G(dest_desc, 0) + d_d1_sstride - d_d1_lb;

  if (d_rank == 2) {
    d_d2_lstride = F90_DIM_LSTRIDE_G(dest_desc, 1);
    d_d2_lb = F90_DIM_LBOUND_G(dest_desc, 1);
    d_d2_sstride = F90_DIM_SSTRIDE_G(dest_desc, 1);
    if (d_d2_sstride != 1 || F90_DIM_SOFFSET_G(dest_desc, 1))
      d_d2_soffset = F90_DIM_SOFFSET_G(dest_desc, 1) + d_d2_sstride - d_d2_lb;
  }

  s1_base = (__LOG8_T *)s1_addr + F90_LBASE_G(s1_desc) +
            s1_d1_lb * s1_d1_lstride + s1_d2_lb * s1_d2_lstride - 1;
  s2_base = (__LOG8_T *)s2_addr + F90_LBASE_G(s2_desc) +
            s2_d1_lb * s2_d1_lstride + s2_d2_lb * s2_d2_lstride - 1;
  dest_base = (__LOG8_T *)dest_addr + F90_LBASE_G(dest_desc) +
              d_d1_lb * d_d1_lstride + d_d2_lb * d_d2_lstride - 1;

  d_d1_offset = d_d1_base = d_d1_soffset * d_d1_lstride;
  d_d1_delta = d_d1_sstride * d_d1_lstride;

  d_d2_offset = d_d2_base = d_d2_soffset * d_d2_lstride;
  d_d2_delta = s1_rank == 2 ? d_d2_sstride * d_d2_lstride : d_d1_delta;

  s1_d1_offset = s1_d1_base = s1_d1_soffset * s1_d1_lstride;
  s1_d1_delta = s1_d1_sstride * s1_d1_lstride;

  s1_d2_offset = s1_d2_base = s1_d2_soffset * s1_d2_lstride;
  s1_d2_delta = s1_rank == 2 ? s1_d2_sstride * s1_d2_lstride : s1_d1_delta;

  s2_d1_offset = s2_d1_base = s2_d1_soffset * s2_d1_lstride;
  s2_d1_delta = s2_d1_sstride * s2_d1_lstride;

  s2_d2_offset = s2_d2_base = s2_d2_soffset * s2_d2_lstride;
  s2_d2_delta = s2_d2_sstride * s2_d2_lstride;

  if (s1_rank == 2) {
    for (k = 0; k < k_extent; k++) {
      d_elem_p = dest_base + d_d1_base + d_d2_offset;
      d_d2_offset += d_d2_delta;
      for (n = 0; n < n_extent; n++) {
        *d_elem_p = 0;
        d_elem_p += d_d1_delta;
      }
    }

    d_d2_offset = d_d2_base;
    for (k = 0; k < k_extent; k++) {
      s2_elem_p = s2_base + s2_d1_base + s2_d2_offset;
      s2_d2_offset += s2_d2_delta;
      s1_d2_offset = s1_d2_base;
      for (m = 0; m < m_extent; m++) {
        s1_elem_p = s1_base + s1_d1_base + s1_d2_offset;
        s1_d2_offset += s1_d2_delta;
        d_elem_p = dest_base + d_d1_base + d_d2_offset;
        for (n = 0; n < n_extent; n++) {
          if ((*s1_elem_p & GET_DIST_MASK_LOG8) &&
              (*s2_elem_p & GET_DIST_MASK_LOG8)) {
            *d_elem_p = GET_DIST_TRUE_LOG8;
          }

          d_elem_p += d_d1_delta;
          s1_elem_p += s1_d1_delta;
        }
        s2_elem_p += s2_d1_delta;
      }
      d_d2_offset += d_d2_delta;
    }
  } else {
    __LOG8_T rslt_tmp;

    s1_base += s1_d1_base;
    s2_base += s2_d1_soffset * s2_d1_lstride;
    dest_offset = d_d1_base;
    for (k = 0; k < k_extent; k++) {
      s1_elem_p = s1_base;
      s2_elem_p = s2_base + s2_d2_base;
      rslt_tmp = 0;
      for (m = 0; m < m_extent; m++) {
        if ((*s1_elem_p & GET_DIST_MASK_LOG8) &&
            (*s2_elem_p & GET_DIST_MASK_LOG8)) {
          rslt_tmp = GET_DIST_TRUE_LOG8;
        }

        s1_elem_p += s1_d1_delta;
        s2_elem_p += s2_d1_delta;
      }
      *(dest_base + dest_offset) = rslt_tmp;
      dest_offset += d_d1_delta;
      s2_d2_base += s2_d2_delta;
    }
  }
}
