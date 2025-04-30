/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* hpf_library, hpf_local_library, and system inquiry routines */

/* FIXME: how much (if any) of this is used/needed */

#include "stdioInterf.h"
#include "fioMacros.h"

static int I8(fetch_int)(void *b, F90_Desc *s)
{
  dtype kind = TYPEKIND(s);
  switch (kind) {
  case __INT1:
    return (int)(*(__INT1_T *)b);
  case __INT2:
    return (int)(*(__INT2_T *)b);
  case __INT4:
    return (int)(*(__INT4_T *)b);
  case __INT8:
    return (int)(*(__INT8_T *)b);
  default:
    __fort_abort("fetch_int: invalid argument type");
    return 0;
  }
}

#ifdef FLANG_QUERY_UNUSED
static int I8(fetch_log)(void *b, F90_Desc *s)
{
  dtype kind = TYPEKIND(s);
  switch (kind) {
  case __LOG1:
    return (*(__LOG1_T *)b & GET_DIST_MASK_LOG1) != 0;
  case __LOG2:
    return (*(__LOG2_T *)b & GET_DIST_MASK_LOG2) != 0;
  case __LOG4:
    return (*(__LOG4_T *)b & GET_DIST_MASK_LOG4) != 0;
  case __LOG8:
    return (*(__LOG8_T *)b & GET_DIST_MASK_LOG8) != 0;
  default:
    __fort_abort("fetch_log: invalid argument type");
    return 0;
  }
}
#endif

static void I8(fetch_vector)(void *ab, F90_Desc *as, __INT_T *vector,
                             int veclen)
{
  __INT_T *la;
  __INT_T i;

  if (F90_RANK_G(as) != 1)
    __fort_abort("fetch_vector: incorrect argument rank");

  for (i = F90_DIM_LBOUND_G(as, 0); --veclen >= 0; ++i) {
    la = I8(__fort_local_address)(ab, as, &i);
    if (la == NULL)
      __fort_abort("fetch_vector: argument inaccessible");
    *vector++ = I8(fetch_int)(la, as);
  }
}

static void I8(store_int)(void *b, F90_Desc *s, __INT_T val)
{
  dtype kind = TYPEKIND(s);
  switch (kind) {
  case __INT1:
    *(__INT1_T *)b = (__INT1_T)val;
    break;
  case __INT2:
    *(__INT2_T *)b = (__INT2_T)val;
    break;
  case __INT4:
    *(__INT4_T *)b = (__INT4_T)val;
    break;
  case __INT8:
    *(__INT8_T *)b = (__INT8_T)val;
    break;
  default:
    __fort_abort("store_int: invalid argument type (integer expected)");
  }
}

static void I8(store_log)(void *b, F90_Desc *s, int val)
{
  dtype kind = TYPEKIND(s);
  switch (kind) {
  case __LOG1:
    *(__LOG1_T *)b = val ? GET_DIST_TRUE_LOG1 : 0;
    break;
  case __LOG2:
    *(__LOG2_T *)b = val ? GET_DIST_TRUE_LOG2 : 0;
    break;
  case __LOG4:
    *(__LOG4_T *)b = val ? GET_DIST_TRUE_LOG4 : 0;
    break;
  case __LOG8:
    *(__LOG8_T *)b = val ? GET_DIST_TRUE_LOG8 : 0;
    break;
  default:
    __fort_abort("store_log: invalid argument type (logical expected)");
  }
}

static void I8(store_element)(void *ab, F90_Desc *as, int index, int val)
{
  __INT_T *la;
  __INT_T i;

  if (F90_RANK_G(as) != 1)
    __fort_abort("store_element: incorrect argument rank");

  i = F90_DIM_LBOUND_G(as, 0) - 1 + index;
  la = I8(__fort_local_address)(ab, as, &i);
  if (la != NULL)
    I8(store_int)(la, as, val);
}

static void I8(store_vector)(void *ab, F90_Desc *as, __INT_T *vector,
                             __INT_T veclen)
{
  __INT_T *la;
  __INT_T i;

  if (F90_RANK_G(as) != 1)
    __fort_abort("store_vector: incorrect argument rank");

  for (i = F90_DIM_LBOUND_G(as, 0); --veclen >= 0; ++i) {
    la = I8(__fort_local_address)(ab, as, &i);
    if (la != NULL)
      I8(store_int)(la, as, *vector);
    ++vector;
  }
}

static void I8(store_vector_int)(void *ab, F90_Desc *as, int *vector,
                                 __INT_T veclen)
{
  __INT_T *la;
  __INT_T i;

  if (F90_RANK_G(as) != 1)
    __fort_abort("store_vector_int: incorrect argument rank");

  for (i = F90_DIM_LBOUND_G(as, 0); --veclen >= 0; ++i) {
    la = I8(__fort_local_address)(ab, as, &i);
    if (la != NULL)
      I8(store_int)(la, as, *vector);
    ++vector;
  }
}

static void ftnstrcpy(char *dst,       /* destination string, blank-filled */
                      size_t len,      /* length of destination space */
                      const char *src) /* null terminated source string  */
{
  char *end = dst + len;
  while (dst < end && *src != '\0')
    *dst++ = *src++;
  while (dst < end)
    *dst++ = ' ';
}

/* FIXME: still used ? */
/* hpf_library mapping inquiry routines */
void ENTFTN(DIST_ALIGNMENT,
            dist_alignment)(void *alignee_b, void *lb, void *ub, void *stride,
                           void *axis_map, void *identity_map, void *dynamic,
                           void *ncopies, F90_Desc *alignee, F90_Desc *lb_s,
                           F90_Desc *ub_s, F90_Desc *stride_s,
                           F90_Desc *axis_map_s, F90_Desc *identity_map_s,
                           F90_Desc *dynamic_s, F90_Desc *ncopies_s)
{
  DECL_DIM_PTRS(ad);
  proc *p;
  procdim *pd;
  __INT_T i, idm, ncp, px, rank, vector[MAXDIMS];

  rank = (F90_TAG_G(alignee) == __DESC) ? F90_RANK_G(alignee) : 0;

  if (ISPRESENT(lb)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TSTRIDE_G(ad) * F90_DPTR_LBOUND_G(ad) +
                  DIST_DPTR_TOFFSET_G(ad) - DIST_DPTR_TLB_G(ad) + 1;
    }
    I8(store_vector)(lb, lb_s, vector, rank);
  }
  if (ISPRESENT(ub)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TSTRIDE_G(ad) * DPTR_UBOUND_G(ad) +
                  DIST_DPTR_TOFFSET_G(ad) - DIST_DPTR_TLB_G(ad) + 1;
    }
    I8(store_vector)(ub, ub_s, vector, rank);
  }
  if (ISPRESENT(stride)) {
    for (i = rank; --i >= 0;) {
      if (DFMT(alignee, i + 1) != DFMT_COLLAPSED) {
        SET_DIM_PTRS(ad, alignee, i);
        vector[i] = DIST_DPTR_TSTRIDE_G(ad);
      } else {
        vector[i] = 0;
      }
    }
    I8(store_vector)(stride, stride_s, vector, rank);
  }
  if (ISPRESENT(axis_map)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TAXIS_G(ad);
    }
    I8(store_vector)(axis_map, axis_map_s, vector, rank);
  }
  if (ISPRESENT(identity_map)) {
    idm = (rank == 0 || rank == F90_RANK_G(DIST_ALIGN_TARGET_G(alignee)));
    for (i = rank; idm && --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      idm = (DIST_DPTR_TAXIS_G(ad) == i + 1 && DIST_DPTR_TSTRIDE_G(ad) == 1 &&
             F90_DPTR_LBOUND_G(ad) == DIST_DPTR_TLB_G(ad) &&
             DPTR_UBOUND_G(ad) == DIST_DPTR_TUB_G(ad));
    }
    I8(store_log)(identity_map, identity_map_s, idm);
  }
  if (ISPRESENT(dynamic)) {
    I8(store_log)(dynamic, dynamic_s, (rank > 0 && F90_FLAGS_G(alignee) & __DYNAMIC));
  }
  if (ISPRESENT(ncopies)) {
    if (rank > 0) {
      p = DIST_DIST_TARGET_G(alignee);
      ncp = 1;
      for (px = 0; px < p->rank; ++px) {
        if (DIST_REPLICATED_G(alignee) >> px & 1) {
          pd = &p->dim[px];
          ncp *= pd->shape;
        }
      }
    } else
      ncp = GET_DIST_TCPUS;
    I8(store_int)(ncopies, ncopies_s, ncp);
  }
}

/* FIXME: still used */
void ENTFTN(DIST_DISTRIBUTIONA, dist_distributiona)(
    void *distributee_b, DCHAR(axis_type), void *axis_info, void *proc_rank,
    void *proc_shape, void *plb, void *pub, void *pstride, void *low_shadow,
    void *high_shadow, F90_Desc *distributee, F90_Desc *axis_type_s,
    F90_Desc *axis_info_s, F90_Desc *proc_rank_s, F90_Desc *proc_shape_s,
    F90_Desc *plb_s, F90_Desc *pub_s, F90_Desc *pstride_s,
    F90_Desc *low_shadow_s, F90_Desc *high_shadow_s DCLEN64(axis_type))
{
  DECL_HDR_PTRS(u);
  DECL_DIM_PTRS(ud);
  DECL_DIM_PTRS(dd);
  proc *p;
  procdim *pd;
  __INT_T i, rank, vector[MAXDIMS];
  const char *src;
  __CLEN_T len;

  if (F90_TAG_G(distributee) == __DESC) {
    u = DIST_ALIGN_TARGET_G(distributee);
    p = DIST_DIST_TARGET_G(distributee);
    rank = F90_RANK_G(u);
  } else {
    u = NULL;
    p = NULL;
    rank = 0;
  }

  if (ISPRESENTC(axis_type)) {
    len = CLEN(axis_type);
    for (i = rank; i > 0; --i) {
      switch (DFMT(u, i)) {
      case DFMT_COLLAPSED:
        src = "COLLAPSED";
        break;
      case DFMT_BLOCK:
      case DFMT_BLOCK_K:
        src = "BLOCK";
        break;
      case DFMT_CYCLIC:
      case DFMT_CYCLIC_K:
        src = "CYCLIC";
        break;
      case DFMT_GEN_BLOCK:
        src = "GEN_BLOCK";
        break;
      case DFMT_INDIRECT:
        src = "INDIRECT";
        break;
      default:
        __fort_abort("DIST_DISTRIBUTION: unsupported dist-format");
      }
      ftnstrcpy(CADR(axis_type) + (i - 1) * len, len, src);
    }
  }
  if (ISPRESENT(axis_info)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DIST_DPTR_BLOCK_G(ud);
    }
    I8(store_vector)(axis_info, axis_info_s, vector, rank);
  }
  if (ISPRESENT(proc_rank)) {
    I8(store_int)(proc_rank, proc_rank_s, p != NULL ? p->rank : 0);
  }
  if (ISPRESENT(proc_shape) && p != NULL) {
    for (i = p->rank; --i >= 0;) {
      pd = &p->dim[i];
      vector[i] = pd->shape;
    }
    I8(store_vector)(proc_shape, proc_shape_s, vector, p->rank);
  }
  if (ISPRESENT(plb)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = 1;
    }
    I8(store_vector)(plb, plb_s, vector, rank);
  }
  if (ISPRESENT(pub)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DIST_DPTR_PSHAPE_G(ud);
    }
    I8(store_vector)(pub, pub_s, vector, rank);
  }
  if (ISPRESENT(pstride)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DIST_DPTR_PSTRIDE_G(ud);
    }
    I8(store_vector)(pstride, pstride_s, vector, rank);
  }

  /* Return low_shadow and high_shadow values for the 'distributee'
     argument.  HPF 2 spec makes no sense where it says these should
     come from the distributee's ultimate align target. */

  rank = (F90_TAG_G(distributee) == __DESC) ? F90_RANK_G(distributee) : 0;

  if (ISPRESENT(low_shadow)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(dd, distributee, i);
      vector[i] = DIST_DPTR_NO_G(dd);
    }
    I8(store_vector)(low_shadow, low_shadow_s, vector, rank);
  }
  if (ISPRESENT(high_shadow)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(dd, distributee, i);
      vector[i] = DIST_DPTR_PO_G(dd);
    }
    I8(store_vector)(high_shadow, high_shadow_s, vector, rank);
  }
}
/* 32 bit CLEN version */
void ENTFTN(DIST_DISTRIBUTION, dist_distribution)(
    void *distributee_b, DCHAR(axis_type), void *axis_info, void *proc_rank,
    void *proc_shape, void *plb, void *pub, void *pstride, void *low_shadow,
    void *high_shadow, F90_Desc *distributee, F90_Desc *axis_type_s,
    F90_Desc *axis_info_s, F90_Desc *proc_rank_s, F90_Desc *proc_shape_s,
    F90_Desc *plb_s, F90_Desc *pub_s, F90_Desc *pstride_s,
    F90_Desc *low_shadow_s, F90_Desc *high_shadow_s DCLEN(axis_type))
{
  ENTFTN(DIST_DISTRIBUTIONA, dist_distributiona)(distributee_b, CADR(axis_type),
         axis_info, proc_rank, proc_shape, plb, pub, pstride, low_shadow,
         high_shadow, distributee, axis_type_s, axis_info_s, proc_rank_s,
         proc_shape_s, plb_s, pub_s, pstride_s, low_shadow_s, high_shadow_s,
         (__CLEN_T)CLEN(axis_type));
}

/* FIXME: not  used */
void ENTFTN(DIST_TEMPLATEA,
            dist_templatea)(void *alignee_b, void *template_rank, void *lb,
                          void *ub, DCHAR(axis_type), void *axis_info,
                          void *number_aligned, void *dynamic,
                          F90_Desc *alignee, F90_Desc *template_rank_s,
                          F90_Desc *lb_s, F90_Desc *ub_s, F90_Desc *axis_type_s,
                          F90_Desc *axis_info_s, F90_Desc *number_aligned_s,
                          F90_Desc *dynamic_s DCLEN64(axis_type))
{
  DECL_HDR_PTRS(u);
  DECL_HDR_PTRS(a);
  DECL_DIM_PTRS(ud);
  proc *p;
  __INT_T i, rank, n_alnd, ux;
  __INT_T alignee_axis[MAXDIMS], vector[MAXDIMS];
  const char *src;
  __CLEN_T len;

  if (F90_TAG_G(alignee) == __DESC) {
    u = DIST_ALIGN_TARGET_G(alignee);
    p = DIST_DIST_TARGET_G(alignee);
    rank = F90_RANK_G(u);
    for (i = rank; --i >= 0;)
      alignee_axis[i] = 0;
    for (i = F90_RANK_G(alignee); --i >= 0;) {
      ux = DIST_DIM_TAXIS_G(alignee, i);
      if (ux > 0)
        alignee_axis[ux - 1] = i + 1;
    }
  } else
    rank = 0;

  if (ISPRESENT(template_rank)) {
    I8(store_int)(template_rank, template_rank_s, rank);
  }
  if (ISPRESENT(lb)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = F90_DPTR_LBOUND_G(ud);
    }
    I8(store_vector)(lb, lb_s, vector, rank);
  }
  if (ISPRESENT(ub)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DPTR_UBOUND_G(ud);
    }
    I8(store_vector)(ub, ub_s, vector, rank);
  }
  if (ISPRESENTC(axis_type)) {
    len = CLEN(axis_type);
    for (i = rank; --i >= 0;) {
      if (alignee_axis[i] > 0)
        src = "NORMAL";
      else if (DIST_SINGLE_G(alignee) >> i & 1)
        src = "SINGLE";
      else
        src = "REPLICATED";
      ftnstrcpy(CADR(axis_type) + i * len, len, src);
    }
  }
  if (ISPRESENT(axis_info)) {
    for (i = rank; --i >= 0;) {
      if (alignee_axis[i] > 0)
        vector[i] = alignee_axis[i];
      else if (DIST_SINGLE_G(alignee) >> i & 1)
        vector[i] = DIST_INFO_G(alignee, i);
      else {
        SET_DIM_PTRS(ud, u, i);
        vector[i] = (DIST_DPTR_PAXIS_G(ud) > 0) ? DIST_DPTR_PSHAPE_G(ud) : 1;
      }
    }
    I8(store_vector)(axis_info, axis_info_s, vector, rank);
  }
  if (ISPRESENT(number_aligned)) {
    if (!(F90_FLAGS_G(u) & __DYNAMIC)) {
      __fort_abort(
          "DIST_TEMPLATE: NUMBER_ALIGNED not supported for static align target");
    }

    n_alnd = 0;
    if (rank > 0) {
      if (u)
        for (a = DIST_NEXT_ALIGNEE_G(u); a != NULL; a = DIST_NEXT_ALIGNEE_G(a)) {
          ++n_alnd;
        }
    }
    I8(store_int)(number_aligned, number_aligned_s, n_alnd);
  }
  if (ISPRESENT(dynamic)) {
    I8(store_log)(dynamic, dynamic_s, rank > 0 && F90_FLAGS_G(u) & __DYNAMIC);
  }
}
/* 32 bit CLEN version */
void ENTFTN(DIST_TEMPLATE,
            dist_template)(void *alignee_b, void *template_rank, void *lb,
                          void *ub, DCHAR(axis_type), void *axis_info,
                          void *number_aligned, void *dynamic,
                          F90_Desc *alignee, F90_Desc *template_rank_s,
                          F90_Desc *lb_s, F90_Desc *ub_s, F90_Desc *axis_type_s,
                          F90_Desc *axis_info_s, F90_Desc *number_aligned_s,
                          F90_Desc *dynamic_s DCLEN(axis_type))
{
  ENTFTN(DIST_TEMPLATEA, dist_templatea)(alignee_b, template_rank, lb, ub,
         CADR(axis_type), axis_info, number_aligned, dynamic, alignee,
         template_rank_s, lb_s, ub_s, axis_type_s, axis_info_s,
         number_aligned_s, dynamic_s, (__CLEN_T)CLEN(axis_type));
}

void ENTFTN(GLOBAL_ALIGNMENT,
            global_alignment)(void *array_b, void *lb, void *ub, void *stride,
                              void *axis_map, void *identity_map, void *dynamic,
                              void *ncopies, F90_Desc *array_s, F90_Desc *lb_s,
                              F90_Desc *ub_s, F90_Desc *stride_s,
                              F90_Desc *axis_map_s, F90_Desc *identity_map_s,
                              F90_Desc *dynamic_s, F90_Desc *ncopies_s)
{
  DECL_HDR_PTRS(alignee);
  DECL_DIM_PTRS(ad);
  proc *p;
  __INT_T i, idm, n, rank, vector[MAXDIMS];

  if (F90_TAG_G(array_s) == __DESC) {
    alignee = DIST_ACTUAL_ARG_G(array_s);
    if (alignee == NULL)
      __fort_abort("GLOBAL_ALIGNMENT: array is not associated"
                  " with global actual argument");
    rank = F90_RANK_G(alignee);
  } else
    rank = 0;

  if (ISPRESENT(lb)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TSTRIDE_G(ad) * F90_DPTR_LBOUND_G(ad) +
                  DIST_DPTR_TOFFSET_G(ad) - DIST_DPTR_TLB_G(ad) + 1;
    }
    I8(store_vector)(lb, lb_s, vector, rank);
  }
  if (ISPRESENT(ub)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TSTRIDE_G(ad) * DPTR_UBOUND_G(ad) +
                  DIST_DPTR_TOFFSET_G(ad) - DIST_DPTR_TLB_G(ad) + 1;
    }
    I8(store_vector)(ub, ub_s, vector, rank);
  }
  if (ISPRESENT(stride)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TSTRIDE_G(ad);
    }
    I8(store_vector)(stride, stride_s, vector, rank);
  }
  if (ISPRESENT(axis_map)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      vector[i] = DIST_DPTR_TAXIS_G(ad);
    }
    I8(store_vector)(axis_map, axis_map_s, vector, rank);
  }
  if (ISPRESENT(identity_map)) {
    idm = (rank == 0 || rank == F90_TAG_G(DIST_ALIGN_TARGET_G(alignee)));
    for (i = rank; idm && --i >= 0;) {
      SET_DIM_PTRS(ad, alignee, i);
      idm = (DIST_DPTR_TAXIS_G(ad) == i + 1 && DIST_DPTR_TSTRIDE_G(ad) == 1 &&
             F90_DPTR_LBOUND_G(ad) == DIST_DPTR_TLB_G(ad) &&
             DPTR_UBOUND_G(ad) == DIST_DPTR_TUB_G(ad));
    }
    I8(store_log)(identity_map, identity_map_s, idm);
  }
  if (ISPRESENT(dynamic)) {
    I8(store_log)(dynamic, dynamic_s, rank > 0 && F90_FLAGS_G(alignee) & __DYNAMIC);
  }
  if (ISPRESENT(ncopies)) {
    if (rank > 0) {
      n = 1;
      p = DIST_DIST_TARGET_G(alignee);
      for (i = p->rank; --i >= 0;) {
        if (DIST_REPLICATED_G(alignee) >> i & 1)
          n *= p->dim[i].shape;
      }
    } else
      n = GET_DIST_TCPUS;
    I8(store_int)(ncopies, ncopies_s, n);
  }
}

void ENTFTN(GLOBAL_DISTRIBUTIONA, global_distributiona)(
    void *array_b, DCHAR(axis_type), void *axis_info, void *proc_rank,
    void *proc_shape, void *plb, void *pub, void *pstride, void *low_shadow,
    void *high_shadow, F90_Desc *array_s, F90_Desc *axis_type_s,
    F90_Desc *axis_info_s, F90_Desc *proc_rank_s, F90_Desc *proc_shape_s,
    F90_Desc *plb_s, F90_Desc *pub_s, F90_Desc *pstride_s,
    F90_Desc *low_shadow_s, F90_Desc *high_shadow_s DCLEN(axis_type))
{
  DECL_HDR_PTRS(u);
  DECL_HDR_PTRS(distributee);
  DECL_DIM_PTRS(ud);
  DECL_DIM_PTRS(dd);
  proc *p;
  procdim *pd;
  __INT_T i, rank, vector[MAXDIMS];
  const char *src;
  __CLEN_T len;

  if (F90_TAG_G(array_s) == __DESC) {
    distributee = DIST_ACTUAL_ARG_G(array_s);
    if (distributee == NULL)
      __fort_abort("GLOBAL_DISTRIBUTION: array is not associated"
                  " with global actual argument");
    u = DIST_ALIGN_TARGET_G(distributee);
    p = DIST_DIST_TARGET_G(distributee);
    rank = F90_RANK_G(u);
  } else {
    distributee = NULL;
    u = NULL;
    p = NULL;
    rank = 0;
  }

  if (ISPRESENTC(axis_type)) {
    len = CLEN(axis_type);
    for (i = rank; i > 0; --i) {
      switch (DFMT(u, i)) {
      case DFMT_COLLAPSED:
        src = "COLLAPSED";
        break;
      case DFMT_BLOCK:
      case DFMT_BLOCK_K:
        src = "BLOCK";
        break;
      case DFMT_CYCLIC:
      case DFMT_CYCLIC_K:
        src = "CYCLIC";
        break;
      case DFMT_GEN_BLOCK:
        src = "GEN_BLOCK";
        break;
      case DFMT_INDIRECT:
        src = "INDIRECT";
        break;
      default:
        __fort_abort("GLOBAL_DISTRIBUTION: unsupported dist-format");
      }
      ftnstrcpy(CADR(axis_type) + (i - 1) * len, len, src);
    }
  }
  if (ISPRESENT(axis_info)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DIST_DPTR_BLOCK_G(ud);
    }
    I8(store_vector)(axis_info, axis_info_s, vector, rank);
  }
  if (ISPRESENT(proc_rank)) {
    I8(store_int)(proc_rank, proc_rank_s, p != NULL ? p->rank : 0);
  }
  if (ISPRESENT(proc_shape) && p != NULL) {
    for (i = p->rank; --i >= 0;) {
      pd = &p->dim[i];
      vector[i] = pd->shape;
    }
    I8(store_vector)(proc_shape, proc_shape_s, vector, p->rank);
  }
  if (ISPRESENT(plb)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = 1;
    }
    I8(store_vector)(plb, plb_s, vector, rank);
  }
  if (ISPRESENT(pub)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DIST_DPTR_PSHAPE_G(ud);
    }
    I8(store_vector)(pub, pub_s, vector, rank);
  }
  if (ISPRESENT(pstride)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DIST_DPTR_PSTRIDE_G(ud);
    }
    I8(store_vector)(pstride, pstride_s, vector, rank);
  }

  /* Return low_shadow and high_shadow values for the 'distributee'
     argument.  HPF 2 spec makes no sense where it says these should
     come from the distributee's ultimate align target. */

  rank = (distributee != NULL && F90_TAG_G(distributee) == __DESC)
             ? F90_RANK_G(distributee)
             : 0;

  if (ISPRESENT(low_shadow)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(dd, distributee, i);
      vector[i] = DIST_DPTR_NO_G(dd);
    }
    I8(store_vector)(low_shadow, low_shadow_s, vector, rank);
  }
  if (ISPRESENT(high_shadow)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(dd, distributee, i);
      vector[i] = DIST_DPTR_PO_G(dd);
    }
    I8(store_vector)(high_shadow, high_shadow_s, vector, rank);
  }
}
/* 32 bit CLEN version */
void ENTFTN(GLOBAL_DISTRIBUTION, global_distribution)(
    void *array_b, DCHAR(axis_type), void *axis_info, void *proc_rank,
    void *proc_shape, void *plb, void *pub, void *pstride, void *low_shadow,
    void *high_shadow, F90_Desc *array_s, F90_Desc *axis_type_s,
    F90_Desc *axis_info_s, F90_Desc *proc_rank_s, F90_Desc *proc_shape_s,
    F90_Desc *plb_s, F90_Desc *pub_s, F90_Desc *pstride_s,
    F90_Desc *low_shadow_s, F90_Desc *high_shadow_s DCLEN(axis_type))
{
  ENTFTN(GLOBAL_DISTRIBUTIONA, global_distributiona)(array_b, CADR(axis_type),
         axis_info, proc_rank, proc_shape, plb, pub, pstride, low_shadow,
         high_shadow, array_s, axis_type_s, axis_info_s, proc_rank_s,
         proc_shape_s, plb_s, pub_s, pstride_s, low_shadow_s, high_shadow_s,
         (__CLEN_T)CLEN(axis_type));
}

void ENTFTN(GLOBAL_TEMPLATEA, global_templatea)(
    void *array_b, void *template_rank, void *lb, void *ub, DCHAR(axis_type),
    void *axis_info, void *number_aligned, void *dynamic, F90_Desc *array_s,
    F90_Desc *template_rank_s, F90_Desc *lb_s, F90_Desc *ub_s,
    F90_Desc *axis_type_s, F90_Desc *axis_info_s, F90_Desc *number_aligned_s,
    F90_Desc *dynamic_s DCLEN64(axis_type))
{
  DECL_HDR_PTRS(u);
  DECL_HDR_PTRS(alignee);
  DECL_HDR_PTRS(a);
  DECL_DIM_PTRS(ud);
  proc *p;
  __INT_T i, rank, n_alnd, ux;
  __INT_T alignee_axis[MAXDIMS], vector[MAXDIMS];
  const char *src;
  __CLEN_T len;

  if (F90_TAG_G(array_s) == __DESC) {
    alignee = DIST_ACTUAL_ARG_G(array_s);
    if (alignee == NULL)
      __fort_abort("GLOBAL_TEMPLATE: array is not associated"
                  " with global actual argument");
    u = DIST_ALIGN_TARGET_G(alignee);
    p = DIST_DIST_TARGET_G(alignee);
    rank = F90_RANK_G(u);
    for (i = rank; --i >= 0;)
      alignee_axis[i] = 0;
    for (i = F90_RANK_G(alignee); --i >= 0;) {
      ux = DIST_DIM_TAXIS_G(alignee, i);
      if (ux > 0)
        alignee_axis[ux - 1] = i + 1;
    }
  } else
    rank = 0;

  if (ISPRESENT(template_rank)) {
    I8(store_int)(template_rank, template_rank_s, rank);
  }
  if (ISPRESENT(lb)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = F90_DPTR_LBOUND_G(ud);
    }
    I8(store_vector)(lb, lb_s, vector, rank);
  }
  if (ISPRESENT(ub)) {
    for (i = rank; --i >= 0;) {
      SET_DIM_PTRS(ud, u, i);
      vector[i] = DPTR_UBOUND_G(ud);
    }
    I8(store_vector)(ub, ub_s, vector, rank);
  }
  if (ISPRESENTC(axis_type)) {
    len = CLEN(axis_type);
    for (i = rank; --i >= 0;) {
      if (alignee_axis[i] > 0)
        src = "NORMAL";
      else if (DIST_SINGLE_G(alignee) >> i & 1)
        src = "SINGLE";
      else
        src = "REPLICATED";
      ftnstrcpy(CADR(axis_type) + i * len, len, src);
    }
  }
  if (ISPRESENT(axis_info)) {
    for (i = rank; --i >= 0;) {
      if (alignee_axis[i] > 0)
        vector[i] = alignee_axis[i];
      else if (DIST_SINGLE_G(alignee) >> i & 1)
        vector[i] = DIST_INFO_G(alignee, i);
      else {
        SET_DIM_PTRS(ud, u, i);
        vector[i] = (DIST_DPTR_PAXIS_G(ud) > 0) ? DIST_DPTR_PSHAPE_G(ud) : 1;
      }
    }
    I8(store_vector)(axis_info, axis_info_s, vector, rank);
  }
  if (ISPRESENT(number_aligned)) {
    n_alnd = 0;
    if (rank > 0) {
      for (a = u; a != NULL; a = DIST_NEXT_ALIGNEE_G(a))
        ++n_alnd;
    }
    I8(store_int)(number_aligned, number_aligned_s, n_alnd);
  }
  if (ISPRESENT(dynamic)) {
    I8(store_log)(dynamic, dynamic_s, rank > 0 && F90_FLAGS_G(u) & __DYNAMIC);
  }
}
/* 32 bit CLEN version */
void ENTFTN(GLOBAL_TEMPLATE, global_template)(
    void *array_b, void *template_rank, void *lb, void *ub, DCHAR(axis_type),
    void *axis_info, void *number_aligned, void *dynamic, F90_Desc *array_s,
    F90_Desc *template_rank_s, F90_Desc *lb_s, F90_Desc *ub_s,
    F90_Desc *axis_type_s, F90_Desc *axis_info_s, F90_Desc *number_aligned_s,
    F90_Desc *dynamic_s DCLEN(axis_type))
{
  ENTFTN(GLOBAL_TEMPLATEA, global_templatea)(array_b, template_rank, lb, ub,
         CADR(axis_type), axis_info, number_aligned, dynamic, array_s,
         template_rank_s, lb_s, ub_s, axis_type_s, axis_info_s,
         number_aligned_s, dynamic_s, (__CLEN_T)CLEN(axis_type));
}

void ENTFTN(GLOBAL_LBOUND, global_lbound)(void *lbound_b, void *array_b,
                                          void *dim_b, F90_Desc *lbound_s,
                                          F90_Desc *array_s, F90_Desc *dim_s)
{
  DECL_HDR_PTRS(g);
  __INT_T i, dim, rank, vector[MAXDIMS];

  if (F90_TAG_G(array_s) == __DESC) {
    g = DIST_ACTUAL_ARG_G(array_s);
    if (g == NULL)
      __fort_abort("GLOBAL_LBOUND: array is not associated"
                  " with global actual argument");
    rank = F90_RANK_G(g);
  } else
    rank = 0;

  if (ISPRESENT(dim_b)) {
    dim = I8(fetch_int)(dim_b, dim_s);
    if (dim < 1 || dim > rank)
      __fort_abort("GLOBAL_LBOUND: invalid dim");
    I8(store_int)(lbound_b, lbound_s, F90_DIM_LBOUND_G(g, dim - 1));
  } else {
    for (i = rank; --i >= 0;)
      vector[i] = F90_DIM_LBOUND_G(g, i);
    I8(store_vector)(lbound_b, lbound_s, vector, rank);
  }
}

void ENTFTN(GLOBAL_SHAPE, global_shape)(void *shape_b, void *source_b,
                                        F90_Desc *shape_s, F90_Desc *source_s)
{
  DECL_HDR_PTRS(g);
  DECL_DIM_PTRS(gd);
  __INT_T i, extent, rank, vector[MAXDIMS];

  if (F90_TAG_G(source_s) == __DESC) {
    g = DIST_ACTUAL_ARG_G(source_s);
    if (g == NULL)
      __fort_abort("GLOBAL_SHAPE: source is not associated with"
                  " global actual argument");
    rank = F90_RANK_G(g);
  } else
    rank = 0;

  for (i = rank; --i >= 0;) {
    SET_DIM_PTRS(gd, g, i);
    extent = F90_DPTR_EXTENT_G(gd);
    if (extent < 0)
      extent = 0;
    vector[i] = extent;
  }
  I8(store_vector)(shape_b, shape_s, vector, rank);
}

void ENTFTN(GLOBAL_SIZE, global_size)(void *size_b, void *array_b, void *dim_b,
                                      F90_Desc *size_s, F90_Desc *array_s,
                                      F90_Desc *dim_s)
{
  DECL_HDR_PTRS(g);
  DECL_DIM_PTRS(gd);
  __INT_T dim, rank, size;

  if (F90_TAG_G(array_s) == __DESC) {
    g = DIST_ACTUAL_ARG_G(array_s);
    if (g == NULL)
      __fort_abort("GLOBAL_SIZE: array is not associated with"
                  " global actual argument");
    rank = F90_RANK_G(g);
    SET_DIM_PTRS(gd, g, 0);
  } else
    rank = 0;

  if (ISPRESENT(dim_b)) {
    dim = I8(fetch_int)(dim_b, dim_s);
    if (dim < 1 || dim > rank)
      __fort_abort("GLOBAL_SIZE: invalid dim");
    SET_DIM_PTRS(gd, g, dim - 1);
    size = F90_DPTR_EXTENT_G(gd);
    if (size < 0)
      size = 0;
  } else if (rank > 0)
    size = F90_GSIZE_G(g);
  else
    size = 1;
  I8(store_int)(size_b, size_s, size);
}

void ENTFTN(GLOBAL_UBOUND, global_ubound)(void *ubound_b, void *array_b,
                                          void *dim_b, F90_Desc *ubound_s,
                                          F90_Desc *array_s, F90_Desc *dim_s)
{
  DECL_HDR_PTRS(g);
  __INT_T i, dim, rank, vector[MAXDIMS];

  if (F90_TAG_G(array_s) == __DESC) {
    g = DIST_ACTUAL_ARG_G(array_s);
    if (g == NULL)
      __fort_abort("GLOBAL_UBOUND: array is not associated with"
                  "global actual argument");
    rank = F90_RANK_G(g);
  } else
    rank = 0;

  if (ISPRESENT(dim_b)) {
    dim = I8(fetch_int)(dim_b, dim_s);
    if (dim < 1 || dim > rank)
      __fort_abort("GLOBAL_UBOUND: invalid dim");
    I8(store_int)(ubound_b, ubound_s, DIM_UBOUND_G(g, dim - 1));
  } else {
    for (i = rank; --i >= 0;)
      vector[i] = DIM_UBOUND_G(g, i);
    I8(store_vector)(ubound_b, ubound_s, vector, rank);
  }
}

void ENTFTN(ABSTRACT_TO_PHYSICAL,
            abstract_to_physical)(void *array_b, void *index_b, void *proc_b,
                                  F90_Desc *array_s, F90_Desc *index_s,
                                  F90_Desc *proc_s)
{
  DECL_HDR_PTRS(g);
  proc *p;
  procdim *pd;
  __INT_T i, index[MAXDIMS], proc;

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("ABSTRACT_TO_PHYSICAL: argument must be array");

  g = DIST_ACTUAL_ARG_G(array_s);
  if (g == NULL)
    __fort_abort("ABSTRACT_TO_PHYSICAL: array is not associated"
                " with global actual argument");

  p = DIST_DIST_TARGET_G(g);

  I8(fetch_vector)(index_b, index_s, index, p->rank);

  proc = p->base;
  for (i = p->rank; --i >= 0;) {
    pd = &p->dim[i];
    if (index[i] < 1 || index[i] > pd->shape)
      __fort_abort("ABSTRACT_TO_PHYSICAL: invalid processor coordinate");
    proc += pd->stride * (index[i] - 1);
  }
  I8(store_int)(proc_b, proc_s, proc);
}

void ENTFTN(PHYSICAL_TO_ABSTRACT,
            physical_to_abstract)(void *array_b, void *proc_b, void *index_b,
                                  F90_Desc *array_s, F90_Desc *proc_s,
                                  F90_Desc *index_s)
{
  DECL_HDR_PTRS(g);
  proc *p;
  procdim *pd;
  __INT_T i, index[MAXDIMS], proc;

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("PHYSICAL_TO_ABSTRACT: argument must be array");

  g = DIST_ACTUAL_ARG_G(array_s);
  if (g == NULL)
    __fort_abort("PHYSICAL_TO_ABSTRACT: array is not associated"
                " with global actual argument");

  p = DIST_DIST_TARGET_G(g);

  proc = I8(fetch_int)(proc_b, proc_s);

  proc -= p->base;
  if (proc < 0 || proc >= p->size)
    __fort_abort("PHYSICAL_TO_ABSTRACT: invalid processor number");

  for (i = 0; i < p->rank; ++i) {
    pd = &p->dim[i];
    RECIP_DIVMOD(&proc, &index[i], proc, pd->shape);
    index[i]++;
  }
  I8(store_vector)(index_b, index_s, index, p->rank);
}

/* Translate local indices to global indices */

void ENTFTN(LOCAL_TO_GLOBAL,
            local_to_global)(void *array_b, void *l_index_b, void *g_index_b,
                             F90_Desc *array_s, F90_Desc *l_index_s,
                             F90_Desc *g_index_s)
{
  DECL_HDR_PTRS(gs);
  DECL_DIM_PTRS(gsd);
  DECL_DIM_PTRS(asd);
  __INT_T i;
  __INT_T index[MAXDIMS];
  __INT_T lboffset, adjindex, cyclenum, cyclepos;

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("LOCAL_TO_GLOBAL: argument must be array");

  gs = DIST_ACTUAL_ARG_G(array_s);
  if (gs == NULL || F90_TAG_G(gs) != __DESC)
    __fort_abort("LOCAL_TO_GLOBAL: array is not associated with"
                " global actual argument");
#if defined(DEBUG)
  if (F90_RANK_G(gs) != F90_RANK_G(array_s))
    __fort_abort("LOCAL_TO_GLOBAL: global vs. local rank mismatch");
#endif

  /* get the local index vector */

  I8(fetch_vector)(l_index_b, l_index_s, index, F90_RANK_G(gs));

  /* translate local array indices to global array indices */

  for (i = 1; i <= F90_RANK_G(gs); ++i) { /* iterate through dimensions */

    SET_DIM_PTRS(asd, array_s, i - 1);
    SET_DIM_PTRS(gsd, gs, i - 1);

    /* index must be within local array bounds */

    if (index[i - 1] < F90_DPTR_LBOUND_G(asd) ||
        index[i - 1] > DPTR_UBOUND_G(asd)) {
      __fort_abort("LOCAL_TO_GLOBAL: index outside local array bounds\n");
    }

    switch (DFMT(gs, i)) {
    case DFMT_CYCLIC:
    case DFMT_CYCLIC_K:

      if (DIST_DPTR_TSTRIDE_G(gsd) != 1) {

        int ii, startblocks, off;
        int elem_per_cycle, elem, my_cycle_lb, my_cycle_ub, first;
        int tstride, abs_tstride, gblock, pcoord, lbound;

        tstride = DIST_DPTR_TSTRIDE_G(gsd);
        abs_tstride = Abs(tstride);
        gblock = DIST_DPTR_BLOCK_G(gsd);
        pcoord = DIST_DPTR_PCOORD_G(gsd);
        lbound = F90_DPTR_LBOUND_G(gsd);
        off = DIST_DPTR_TOFFSET_G(gsd);
        first = (lbound * tstride + off) - 1;
        elem = first + 1;

        if (tstride < 0) {

          int start_cpu, ext, text, partialblocks, tlb, tub, cpus;

          tlb = DIST_DPTR_TLB_G(gsd);

          tub = DIST_DPTR_TUB_G(gsd);

          ext = elem - tlb + 1;

          text = tub - tlb + 1;

          cpus = Min(DIST_DPTR_PSHAPE_G(gsd), Ceil(text, gblock));

          elem_per_cycle = (gblock * cpus);

          partialblocks = (ext % elem_per_cycle);

          if (!partialblocks) {
            start_cpu = cpus - 1;
            startblocks = gblock * Abs(pcoord - start_cpu);
          } else if (partialblocks <= gblock) {
            start_cpu = 0;
            startblocks = partialblocks * Abs(pcoord - start_cpu);
            if (!startblocks)
              startblocks = partialblocks - gblock;
          } else {

            RECIP_DIV(&start_cpu, partialblocks, DIST_DPTR_BLOCK_G(gsd));

            if (start_cpu < 0)
              start_cpu += cpus;
            else if (start_cpu >= cpus)
              start_cpu -= cpus;

            startblocks = Abs(pcoord - start_cpu);
            startblocks *= (partialblocks - gblock);
          }

          elem = tub - elem + 1;
          first = elem - 1;

        } else {
          elem_per_cycle = DIST_DPTR_CYCLE_G(gsd);
          startblocks = pcoord * gblock;
        }

        my_cycle_lb = (lbound + startblocks);

        if (my_cycle_lb > lbound) {
          while (elem > my_cycle_lb)
            my_cycle_lb += elem_per_cycle;
        }

        my_cycle_ub = my_cycle_lb + (gblock - 1);

        elem -= abs_tstride;
        for (ii = F90_DPTR_LBOUND_G(asd); ii <= index[i - 1];) {
          if (elem > my_cycle_ub) {
            my_cycle_lb += elem_per_cycle;
            my_cycle_ub += elem_per_cycle;
          } else
            elem += abs_tstride;

          if (elem >= my_cycle_lb && elem <= my_cycle_ub) {
            ++ii;
          }
        }

        index[i - 1] =
            (elem - first) / abs_tstride + (elem - first) % abs_tstride;

        break;
      }

      if (DIST_DPTR_OLB_G(gsd) == F90_DPTR_LBOUND_G(gsd)) { /* First element */
        lboffset = 0;
      } else {
        lboffset = 0;
      }
      adjindex = index[i - 1] - F90_DPTR_LBOUND_G(asd) + lboffset;
      RECIP_DIVMOD(&cyclenum, &cyclepos, adjindex, DIST_DPTR_BLOCK_G(gsd));
      index[i - 1] = cyclenum * DIST_DPTR_CYCLE_G(gsd) + cyclepos +
                     DIST_DPTR_OLB_G(gsd) - lboffset;

      break;

    default: /* block */
      index[i - 1] += DIST_DPTR_OLB_G(gsd) - F90_DPTR_LBOUND_G(asd);
    }
  }

  /* return the global index vector */

  I8(store_vector)(g_index_b, g_index_s, index, F90_RANK_G(gs));
}

/* Translate global indices to local indices */

void ENTFTN(GLOBAL_TO_LOCAL,
            global_to_local)(void *array_b, void *g_index_b, void *l_index_b,
                             void *local_b, void *ncopies_b, void *procs_b,
                             F90_Desc *array_s, F90_Desc *g_index_s,
                             F90_Desc *l_index_s, F90_Desc *local_s,
                             F90_Desc *ncopies_s, F90_Desc *procs_s)
{
  DECL_DIM_PTRS(asd); /* local array dimensions */
  DECL_HDR_PTRS(gs);  /* global section */
  DECL_DIM_PTRS(gsd); /* global section dimensions */
  repl_t repl; /* replication descriptor */
  __INT_T i, j, local, lof, procno;
  __INT_T *procs;
  __INT_T gindex[MAXDIMS], lindex[MAXDIMS], pcoord[MAXDIMS];

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("GLOBAL_TO_LOCAL: argument must be array");
  gs = DIST_ACTUAL_ARG_G(array_s);
  if (gs == NULL || F90_TAG_G(gs) != __DESC)
    __fort_abort("GLOBAL_TO_LOCAL: array is not associated with"
                " global actual argument");
#if defined(DEBUG)
  if (F90_RANK_G(gs) != F90_RANK_G(array_s))
    __fort_abort("GLOBAL_TO_LOCAL: global vs. local rank mismatch");
#endif

  /* get the global index vector */

  I8(fetch_vector)(g_index_b, g_index_s, gindex, F90_RANK_G(gs));

  /* check if element is local */

  local = I8(__fort_islocal)(gs, gindex);

  if (local && ISPRESENT(l_index_b)) {
    for (i = F90_RANK_G(gs); i > 0; --i) {
      SET_DIM_PTRS(asd, array_s, i - 1);
      SET_DIM_PTRS(gsd, gs, i - 1);

      switch (DFMT(gs, i)) {
      case DFMT_CYCLIC:
      case DFMT_CYCLIC_K: {

        __INT_T aolb, aoub;

        aolb = DIST_DPTR_OLB_G(asd);
        aoub = DIST_DPTR_OUB_G(asd);

        /* compute local offset for cyclic distribution */

        j = DIST_DPTR_TSTRIDE_G(gsd) * gindex[i - 1] + DIST_DPTR_TOFFSET_G(gsd) -
            DIST_DPTR_CLB_G(gsd);
        j = Abs(j);
        RECIP_DIV(&j, j, DIST_DPTR_CYCLE_G(gsd));
        lof = j * DIST_DPTR_COFSTR_G(gsd);

        lindex[i - 1] = F90_DPTR_SSTRIDE_G(gsd) * gindex[i - 1] +
                        F90_DPTR_SOFFSET_G(gsd) - lof -
                        (DIST_DPTR_OLB_G(gsd) - aolb);

        while (lindex[i - 1] > aoub) {
          lindex[i - 1] -= (DIST_DPTR_CYCLE_G(gsd) - DIST_DPTR_BLOCK_G(gsd));
        }

        while (lindex[i - 1] < aolb) {
          lindex[i - 1] += (aoub - aolb + 1);
        }
        break;
      }

      default:
        /* block or unmapped: subtract the difference between
           global and local owned bounds */
        lindex[i - 1] =
            gindex[i - 1] - (DIST_DPTR_OLB_G(gsd) - DIST_DPTR_OLB_G(asd));
      }
    }
    I8(store_vector)(l_index_b, l_index_s, lindex, F90_RANK_G(gs));
  }

  if (ISPRESENT(local_b))
    I8(store_log)(local_b, local_s, local);

  /*  if needed, get replication info */

  if (ISPRESENT(ncopies_b) || ISPRESENT(procs_b))
    I8(__fort_describe_replication)(gs, &repl);

  if (ISPRESENT(ncopies_b))
    I8(store_int)(ncopies_b, ncopies_s, repl.ncopies);

  if (ISPRESENT(procs_b)) {
    procno = I8(__fort_owner)(gs, gindex);
    if (repl.ncopies == 1)
      I8(store_vector)(procs_b, procs_s, &procno, 1);
    else {
      procs = (__INT_T *)__fort_malloc(repl.ncopies * sizeof(__INT_T));
      for (i = repl.ndim; --i >= 0;)
        pcoord[i] = 0;
      i = j = 0;
      while (j < repl.ndim) {
        if (pcoord[j] < repl.pcnt[j]) {
          procs[i++] = procno;
          procno += repl.pstr[j];
          ++pcoord[j];
          j = 0;
        } else {
          procno -= repl.pcnt[j] * repl.pstr[j];
          pcoord[j++] = 0;
        }
      }
#if defined(DEBUG)
      if (i != repl.ncopies)
        __fort_abort("GLOBAL_TO_LOCAL: replication info incorrect");
#endif
      I8(store_vector)(procs_b, procs_s, procs, repl.ncopies);
      __fort_free(procs);
    }
  }
}

/* Return the number of non-empty blocks in all or a specified dimension. */

void ENTFTN(LOCAL_BLKCNT, local_blkcnt)(void *blkcnt_b, void *array_b,
                                        void *dim_b, void *proc_b,
                                        F90_Desc *blkcnt_s, F90_Desc *array_s,
                                        F90_Desc *dim_s, F90_Desc *proc_s)
{
  DECL_HDR_PTRS(gs);
  DECL_DIM_PTRS(gsd);
  __INT_T dim, proc;
  __INT_T blkcnt[MAXDIMS];
  __INT_T cl, cn, il, iu;

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("LOCAL_BLKCNT: argument must be array");
  if ((gs = DIST_ACTUAL_ARG_G(array_s)) == NULL)
    __fort_abort("LOCAL_BLKCNT: array is not associated with global"
                " actual argument");

  if (ISPRESENT(dim_b)) {
    dim = I8(fetch_int)(dim_b, dim_s);
    if (dim < 1 || dim > F90_RANK_G(gs))
      __fort_abort("LOCAL_BLKCNT: invalid dim");
  } else
    dim = 0;

  if (ISPRESENT(proc_b)) {
    if ((proc = I8(fetch_int)(proc_b, proc_s)) < 0 || proc >= GET_DIST_TCPUS)
      __fort_abort("LOCAL_BLKCNT: invalid proc");
    if (proc != GET_DIST_LCPU)
      __fort_abort("LOCAL_BLKCNT: proc .ne. my_processor() unsupported");
  } else
    proc = GET_DIST_LCPU;

  if (dim != 0) {

    /* compute blkcnt for specified dimension */

    blkcnt[0] = 0;
    if (~F90_FLAGS_G(gs) & __OFF_TEMPLATE) {
      I8(__fort_cycle_bounds)(gs);
      SET_DIM_PTRS(gsd, gs, dim - 1);
      for (cl = DIST_DPTR_CL_G(gsd), cn = DIST_DPTR_CN_G(gsd); --cn >= 0;
           cl += DIST_DPTR_CS_G(gsd))
        if (I8(__fort_block_bounds)(gs, dim, cl, &il, &iu) > 0)
          blkcnt[0]++;
    }
    I8(store_int)(blkcnt_b, blkcnt_s, blkcnt[0]);
  } else {

    /* compute blkcnt for all dimensions */

    for (dim = F90_RANK_G(gs); dim > 0; --dim)
      blkcnt[dim - 1] = 0;
    if (~F90_FLAGS_G(gs) & __OFF_TEMPLATE) {
      I8(__fort_cycle_bounds)(gs);
      for (dim = F90_RANK_G(gs); dim > 0; --dim) {
        SET_DIM_PTRS(gsd, gs, dim - 1);
        for (cl = DIST_DPTR_CL_G(gsd), cn = DIST_DPTR_CN_G(gsd); --cn >= 0;
             cl += DIST_DPTR_CS_G(gsd))
          if (I8(__fort_block_bounds)(gs, dim, cl, &il, &iu) > 0)
            blkcnt[dim - 1]++;
      }
    }
    I8(store_vector)(blkcnt_b, blkcnt_s, blkcnt, F90_RANK_G(gs));
  }
}

/* Return the lower indices of all non-empty blocks. */

void ENTFTN(LOCAL_LINDEX, local_lindex)(void *lindex_b, void *array_b,
                                        void *dim_b, void *proc_b,
                                        F90_Desc *lindex_s, F90_Desc *array_s,
                                        F90_Desc *dim_s, F90_Desc *proc_s)
{
  DECL_HDR_PTRS(gs);
  DECL_DIM_PTRS(gsd);
  __INT_T dim;
  __INT_T proc;
  __INT_T blkcnt, cl, cn, il, iu;

  /* check array argument */

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("LOCAL_LINDEX: argument must be array");
  if ((gs = DIST_ACTUAL_ARG_G(array_s)) == NULL)
    __fort_abort("LOCAL_LINDEX: array is not associated with global"
                " actual argument");

  /* check dim argument */

  dim = I8(fetch_int)(dim_b, dim_s);
  if (dim < 1 || dim > F90_RANK_G(gs))
    __fort_abort("LOCAL_LINDEX: invalid dim argument");

  /* check proc argument */

  if (ISPRESENT(proc_b)) {
    proc = I8(fetch_int)(proc_b, proc_s);
    if (proc < 0 || proc >= GET_DIST_TCPUS)
      __fort_abort("LOCAL_LINDEX: invalid proc argument");
    if (proc != GET_DIST_LCPU)
      __fort_abort("LOCAL_LINDEX: proc .ne. my_processor() unsupported");
  } else
    proc = GET_DIST_LCPU;

  /* compute lower indices of all non-empty blocks */

  if (~F90_FLAGS_G(gs) & __OFF_TEMPLATE) {
    I8(__fort_cycle_bounds)(gs);
    SET_DIM_PTRS(gsd, gs, dim - 1);
    blkcnt = 0;
    for (cl = DIST_DPTR_CL_G(gsd), cn = DIST_DPTR_CN_G(gsd); --cn >= 0;
         cl += DIST_DPTR_CS_G(gsd)) {
      if (I8(__fort_block_bounds)(gs, dim, cl, &il, &iu) > 0) {

        DECL_DIM_PTRS(asd);

        SET_DIM_PTRS(asd, array_s, dim - 1);

        switch (DFMT(gs, dim)) {
        case DFMT_CYCLIC:
        case DFMT_CYCLIC_K: {

          __INT_T aolb, aoub, j, lof;

          aolb = DIST_DPTR_OLB_G(asd);
          aoub = DIST_DPTR_OUB_G(asd);

          /* compute local offset for cyclic distribution */

          j = DIST_DPTR_TSTRIDE_G(gsd) * il + DIST_DPTR_TOFFSET_G(gsd) -
              DIST_DPTR_CLB_G(gsd);
          j = Abs(j);
          RECIP_DIV(&j, j, DIST_DPTR_CYCLE_G(gsd));
          lof = j * DIST_DPTR_COFSTR_G(gsd);

          il = F90_DPTR_SSTRIDE_G(gsd) * il + F90_DPTR_SOFFSET_G(gsd) - lof -
               (DIST_DPTR_OLB_G(gsd) - aolb);

          while (il > aoub) {
            il -= (DIST_DPTR_CYCLE_G(gsd) - DIST_DPTR_BLOCK_G(gsd));
          }

          while (il < aolb) {
            il += (aoub - aolb + 1);
          }
          break;
        }

        default:
          /* block or unmapped: subtract the difference between
             global and local owned bounds */
          il -= (DIST_DPTR_OLB_G(gsd) - DIST_DPTR_OLB_G(asd));
        }
        I8(store_element)(lindex_b, lindex_s, ++blkcnt, il);
      }
    }
  }
}

/* Return the upper indices of all non-empty blocks */

void ENTFTN(LOCAL_UINDEX, local_uindex)(void *uindex_b, void *array_b,
                                        void *dim_b, void *proc_b,
                                        F90_Desc *uindex_s, F90_Desc *array_s,
                                        F90_Desc *dim_s, F90_Desc *proc_s)
{
  DECL_HDR_PTRS(gs);
  DECL_DIM_PTRS(gsd);
  __INT_T dim;
  __INT_T proc;
  __INT_T blkcnt, cl, cn, il, iu;

  /* check array argument */

  if (F90_TAG_G(array_s) != __DESC)
    __fort_abort("LOCAL_UINDEX: argument must be array");
  if ((gs = DIST_ACTUAL_ARG_G(array_s)) == NULL)
    __fort_abort("LOCAL_UINDEX: array is not associated with global"
                " actual argument");

  /* check dim argument */

  dim = I8(fetch_int)(dim_b, dim_s);
  if (dim < 1 || dim > F90_RANK_G(gs))
    __fort_abort("LOCAL_UINDEX: invalid dim argument");

  /* check proc argument */

  if (ISPRESENT(proc_b)) {
    proc = I8(fetch_int)(proc_b, proc_s);
    if (proc < 0 || proc >= GET_DIST_TCPUS)
      __fort_abort("LOCAL_UINDEX: invalid proc argument");
    if (proc != GET_DIST_LCPU)
      __fort_abort("LOCAL_UINDEX: proc .ne. my_processor() unsupported");
  } else
    proc = GET_DIST_LCPU;

  /* compute upper indices of all non-empty blocks */

  if (~F90_FLAGS_G(gs) & __OFF_TEMPLATE) {
    I8(__fort_cycle_bounds)(gs);
    SET_DIM_PTRS(gsd, gs, dim - 1);
    blkcnt = 0;
    for (cl = DIST_DPTR_CL_G(gsd), cn = DIST_DPTR_CN_G(gsd); --cn >= 0;
         cl += DIST_DPTR_CS_G(gsd)) {
      if (I8(__fort_block_bounds)(gs, dim, cl, &il, &iu) > 0) {
        DECL_DIM_PTRS(asd);

        SET_DIM_PTRS(asd, array_s, dim - 1);

        switch (DFMT(gs, dim)) {
        case DFMT_CYCLIC:
        case DFMT_CYCLIC_K: {

          __INT_T aolb, aoub, j, lof;

          aolb = DIST_DPTR_OLB_G(asd);
          aoub = DIST_DPTR_OUB_G(asd);

          /* compute local offset for cyclic distribution */

          j = DIST_DPTR_TSTRIDE_G(gsd) * iu + DIST_DPTR_TOFFSET_G(gsd) -
              DIST_DPTR_CLB_G(gsd);
          j = Abs(j);
          RECIP_DIV(&j, j, DIST_DPTR_CYCLE_G(gsd));
          lof = j * DIST_DPTR_COFSTR_G(gsd);

          iu = F90_DPTR_SSTRIDE_G(gsd) * iu + F90_DPTR_SOFFSET_G(gsd) - lof -
               (DIST_DPTR_OLB_G(gsd) - aolb);

          while (iu > aoub) {
            iu -= (DIST_DPTR_CYCLE_G(gsd) - DIST_DPTR_BLOCK_G(gsd));
          }

          while (iu < aolb) {
            iu += (aoub - aolb + 1);
          }
          break;
        }

        default:
          /* block or unmapped: subtract the difference between
             global and local owned bounds */
          iu -= (DIST_DPTR_OLB_G(gsd) - DIST_DPTR_OLB_G(asd));
        }
        I8(store_element)(uindex_b, uindex_s, ++blkcnt, iu);
      }
    }
  }
}

/* system inquiry routines */

void ENTFTN(PROCESSORS_SHAPE, processors_shape)(__INT_T *shape,
                                                F90_Desc *shape_s)
{
  I8(store_vector_int)(shape, shape_s, GET_DIST_TCPUS_ADDR, 1);
}

#ifndef DESC_I8

__INT_T
ENTFTN(MY_PROCESSOR, my_processor)() { return GET_DIST_LCPU; }

__INT_T
ENTFTN(MYPROCNUM, myprocnum)() { return GET_DIST_LCPU; }

int
__fort_nprocs()
{
  return GET_DIST_TCPUS;
}

__INT_T
ENTFTN(NPROCS, nprocs)() { return GET_DIST_TCPUS; }

__INT_T
ENTFTN(NUMBER_OF_PROCESSORS, number_of_processors)
(__INT_T *dim, __INT_T *szdim)
{
  int d, np;

  np = GET_DIST_TCPUS;
  if (ISPRESENT(dim)) {
    d = __fort_varying_int(dim, szdim);
    if (d != 1)
      np = 1;
  }
  return np;
}

__INT8_T
ENTFTN(KNUMBER_OF_PROCESSORS, knumber_of_processors)
(__INT_T *dim, __INT_T *szdim)
{

  /* 
   * -i8 variant of NUMBER_OF_PROCESSORS
   */

  int d, np;

  np = GET_DIST_TCPUS;
  if (ISPRESENT(dim)) {
    d = __fort_varying_int(dim, szdim);
    if (d != 1)
      np = 1;
  }
  return np;
}

__INT_T
ENTFTN(PROCESSORS_RANK, processors_rank)() { return 1; }

__INT8_T
ENTFTN(KPROCESSORS_RANK, kprocessors_rank)()
{

  /* 
   * -i8 variant of PROCESSORS_RANK
   */

  return 1;
}

/* Translate processor number to processor grid coordinates.
   rank and shape describe the processor grid.  The processor
   number given by procnum is translated to grid coordinates returned
   in coord.  Grid coordinates are integers between 1 and the size of
   the corresponding grid dimension.  If the processor number is
   outside the bounds of the processor grid, zeroes are returned in
   coord.  */

void
__fort_procnum_to_coord(int procnum, int rank, __INT_T *shape, __INT_T *coord)
{
  int i, m;

  if (procnum >= 0) {
    for (i = 0; i < rank; ++i) {
      if (shape[i] <= 0)
        __fort_abort("PROCNUM_TO_COORD: invalid processor shape");
      m = procnum / shape[i];
      coord[i] = procnum - m * shape[i] + 1;
      procnum = m;
    }
  }
  if (procnum != 0) {
    for (i = rank; --i >= 0;)
      coord[i] = 0;
  }
}

void ENTFTN(PROCNUM_TO_COORD, procnum_to_coord)(__INT_T *procnum, __INT_T *rank,
                                                __INT_T *shape, __INT_T *coord)
{
  __fort_procnum_to_coord(*procnum, *rank, shape, coord);
}

/* Translate processor grid coordinates to processor number.
   rank and shape describe the processor grid.  The processor grid
   coordinates in coord are translated to a processor number.
   Grid coordinates are integers between 1 and the size of the
   corresponding grid dimension.  If the coordinates are outside the
   bounds of the processor grid, -1 is returned.  */

int
__fort_coord_to_procnum(__INT_T rank, __INT_T *shape, __INT_T *coord)
{
  int i, m, p;

  m = 1;
  p = 0;
  for (i = 0; i < rank; ++i) {
    if (shape[i] <= 0)
      __fort_abort("COORD_TO_PROCNUM: invalid processor shape");
    if (coord[i]<1 | coord[i]> shape[i])
      return -1;
    p += (coord[i] - 1) * m;
    m *= shape[i];
  }
  return p;
}

__INT_T
ENTFTN(COORD_TO_PROCNUM, coord_to_procnum)
(__INT_T *rank, __INT_T *shape, __INT_T *coord)
{
  return __fort_coord_to_procnum(*rank, shape, coord);
}

#endif
