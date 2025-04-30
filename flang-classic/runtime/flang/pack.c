/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* pack/unpack intrinsics */

#include "stdioInterf.h"
#include "fioMacros.h"

extern void (*__fort_scalar_copy[__NTYPES])(void *rp, const void *sp, int len);

static int I8(next_index)(__INT_T *index, F90_Desc *s)
{
  __INT_T i;

  for (i = 0; i < F90_RANK_G(s); i++) {
    index[i]++;
    if (index[i] <= DIM_UBOUND_G(s, i))
      return 1; /* keep going */
    index[i] = F90_DIM_LBOUND_G(s, i);
  }
  return 0; /* finished */
}

/* pack, optional vector arg present.  pack masked elements of array
   into result and fill remainder of result with corresponding
   elements of vector */

void ENTFTN(PACK, pack)(void *rb,         /* result base */
                        void *ab,         /* array base */
                        void *mb,         /* mask base */
                        void *vb,         /* vector base */
                        F90_Desc *result, /* result descriptor */
                        F90_Desc *array,  /* array descriptor */
                        F90_Desc *mask,   /* mask descriptor */
                        F90_Desc *vector) /* vector descriptor */
{
  char *la, *rf, *vf;
  __INT_T rindex;
  __INT_T vindex;
  __INT_T aindex[MAXDIMS];
  __INT_T mindex[MAXDIMS];
  __BIGREAL_T tmp[4];
  __INT_T mlen;
  __INT_T i, mask_is_array, more_array, more_vector, mval;

  if (result == NULL || F90_TAG_G(result) != __DESC)
    __fort_abort("PACK: invalid result descriptor");

  if (vector == NULL || F90_TAG_G(vector) != __DESC)
    __fort_abort("PACK: invalid vector descriptor");

  if (F90_GSIZE_G(result) == 0 || F90_GSIZE_G(vector) == 0)
    return;

  rf = (char *)rb + DIST_SCOFF_G(result) * F90_LEN_G(result);
  vf = (char *)vb + DIST_SCOFF_G(vector) * F90_LEN_G(vector);

  rindex = F90_DIM_LBOUND_G(result, 0);
  vindex = F90_DIM_LBOUND_G(vector, 0);
  for (i = F90_RANK_G(array); --i >= 0;) {
    aindex[i] = F90_DIM_LBOUND_G(array, i);
  }

  if (ISSCALAR(mask)) {
    mlen = GET_DIST_SIZE_OF(TYPEKIND(mask));
    mval = I8(__fort_varying_log)(mb, &mlen);
    mask_is_array = 0;
  } else if (F90_TAG_G(mask) == __DESC) {
    for (i = F90_RANK_G(mask); --i >= 0;)
      mindex[i] = F90_DIM_LBOUND_G(mask, i);
    mask_is_array = 1;
  } else
    __fort_abort("PACK: invalid mask descriptor");

  more_array = more_vector = 1;
  while (more_array & more_vector) {

    /* get mask value */

    if (mask_is_array) {
      I8(__fort_get_scalar)(tmp, mb, mask, mindex);
      switch (F90_KIND_G(mask)) {
      case __LOG1:
        mval = (*(__LOG1_T *)tmp & GET_DIST_MASK_LOG1) != 0;
        break;
      case __LOG2:
        mval = (*(__LOG2_T *)tmp & GET_DIST_MASK_LOG2) != 0;
        break;
      case __LOG4:
        mval = (*(__LOG4_T *)tmp & GET_DIST_MASK_LOG4) != 0;
        break;
      case __LOG8:
        mval = (*(__LOG8_T *)tmp & GET_DIST_MASK_LOG8) != 0;
        break;
      case __INT1:
        mval = (*(__INT1_T *)tmp & GET_DIST_MASK_INT1) != 0;
        break;
      case __INT2:
        mval = (*(__INT2_T *)tmp & GET_DIST_MASK_INT2) != 0;
        break;
      case __INT4:
        mval = (*(__INT4_T *)tmp & GET_DIST_MASK_INT4) != 0;
        break;
      case __INT8:
        mval = (*(__INT8_T *)tmp & GET_DIST_MASK_INT8) != 0;
        break;
      default:
        __fort_abort("PACK: unknown mask type");
      }
      more_array &= I8(next_index)(mindex, mask);
    }

    /* if mask is true, store the corresponding array element into
       the next result element and also advance to the next vector
       element. */

    if (mval) {
      la = I8(__fort_local_address)(rf, result, &rindex);
      if (la == NULL)
        la = (char *)tmp;
      I8(__fort_get_scalar)(la, ab, array, aindex);
      more_vector &= I8(next_index)(&rindex, result);
      more_vector &= I8(next_index)(&vindex, vector);
    }
    more_array &= I8(next_index)(aindex, array);
  }

  /* if there are fewer masked elements than result elements, fill
     the remainder of the result with the corresponding vector
     elements. */

  while (more_vector) {
    la = I8(__fort_local_address)(rf, result, &rindex);
    if (la == NULL)
      la = (char *)tmp;
    I8(__fort_get_scalar)(la, vf, vector, &vindex);
    more_vector &= I8(next_index)(&rindex, result);
    more_vector &= I8(next_index)(&vindex, vector);
  }
}

void ENTFTN(PACKCA, packca)(DCHAR(rb),        /* result char base */
                          DCHAR(ab),        /* array char base */
                          void *mb,         /* mask base */
                          DCHAR(vb),        /* vector char base */
                          F90_Desc *result, /* result descriptor */
                          F90_Desc *array,  /* array descriptor */
                          F90_Desc *mask,   /* mask descriptor */
                          F90_Desc *vector  /* vector descriptor */
                          DCLEN64(rb)         /* result char len */
                          DCLEN64(ab)         /* array char len */
                          DCLEN64(vb))        /* vector char len */
{
  ENTFTN(PACK,pack)(CADR(rb), CADR(ab), mb, CADR(vb),
		      result, array, mask, vector);
}
/* 32 bit CLEN version */
void ENTFTN(PACKC, packc)(DCHAR(rb),        /* result char base */
                          DCHAR(ab),        /* array char base */
                          void *mb,         /* mask base */
                          DCHAR(vb),        /* vector char base */
                          F90_Desc *result, /* result descriptor */
                          F90_Desc *array,  /* array descriptor */
                          F90_Desc *mask,   /* mask descriptor */
                          F90_Desc *vector  /* vector descriptor */
                          DCLEN(rb)         /* result char len */
                          DCLEN(ab)         /* array char len */
                          DCLEN(vb))        /* vector char len */
{
  ENTFTN(PACKCA, packca)(CADR(rb), CADR(ab), mb, CADR(vb), result, array, mask,
            vector, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab), (__CLEN_T)CLEN(vb));
}

/* pack, optional vector arg absent.  pack masked elements of array
   into result. */

void ENTFTN(PACKZ, packz)(void *rb,         /* result base */
                          void *ab,         /* array base */
                          void *mb,         /* mask base */
                          F90_Desc *result, /* result descriptor */
                          F90_Desc *array,  /* array descriptor */
                          F90_Desc *mask)   /* mask descriptor */
{
  char *la, *rf;
  __INT_T rindex;
  __INT_T aindex[MAXDIMS];
  __INT_T mindex[MAXDIMS];
  __BIGREAL_T tmp[4];
  __INT_T mlen;
  __INT_T i, mask_is_array, more, mval;

  if (result == NULL || F90_TAG_G(result) != __DESC)
    __fort_abort("PACK: invalid result descriptor");

  if (F90_GSIZE_G(result) == 0)
    return;

  rf = (char *)rb + DIST_SCOFF_G(result) * F90_LEN_G(result);

  rindex = F90_DIM_LBOUND_G(result, 0);

  for (i = F90_RANK_G(array); --i >= 0;)
    aindex[i] = F90_DIM_LBOUND_G(array, i);

  if (ISSCALAR(mask)) {
    mlen = GET_DIST_SIZE_OF(TYPEKIND(mask));
    mval = I8(__fort_varying_log)(mb, &mlen);
    if (!mval)
      return;
    mask_is_array = 0;
  } else if (F90_TAG_G(mask) == __DESC) {
    for (i = F90_RANK_G(mask); --i >= 0;)
      mindex[i] = F90_DIM_LBOUND_G(mask, i);
    mask_is_array = 1;
  } else
    __fort_abort("PACK: invalid mask descriptor");

  more = 1;
  while (more) {

    /* get mask value */

    if (mask_is_array) {
      I8(__fort_get_scalar)(tmp, mb, mask, mindex);
      switch (F90_KIND_G(mask)) {
      case __LOG1:
        mval = (*(__LOG1_T *)tmp & GET_DIST_MASK_LOG1) != 0;
        break;
      case __LOG2:
        mval = (*(__LOG2_T *)tmp & GET_DIST_MASK_LOG2) != 0;
        break;
      case __LOG4:
        mval = (*(__LOG4_T *)tmp & GET_DIST_MASK_LOG4) != 0;
        break;
      case __LOG8:
        mval = (*(__LOG8_T *)tmp & GET_DIST_MASK_LOG8) != 0;
        break;
      case __INT1:
        mval = (*(__INT1_T *)tmp & GET_DIST_MASK_INT1) != 0;
        break;
      case __INT2:
        mval = (*(__INT2_T *)tmp & GET_DIST_MASK_INT2) != 0;
        break;
      case __INT4:
        mval = (*(__INT4_T *)tmp & GET_DIST_MASK_INT4) != 0;
        break;
      case __INT8:
        mval = (*(__INT8_T *)tmp & GET_DIST_MASK_INT8) != 0;
        break;
      default:
        __fort_abort("PACK: unknown mask type");
      }
      more &= I8(next_index)(mindex, mask);
    }

    /* if mask is true, store the corresponding array element into
       the next result element. */

    if (mval) {
      la = I8(__fort_local_address)(rf, result, &rindex);
      if (la == NULL)
        la = (char *)tmp;
      I8(__fort_get_scalar)(la, ab, array, aindex);
      more &= I8(next_index)(&rindex, result);
    }
    more &= I8(next_index)(aindex, array);
  }
}

void ENTFTN(PACKZCA, packzca)(DCHAR(rb),        /* result char base */
                            DCHAR(ab),        /* array char base */
                            void *mb,         /* mask base */
                            F90_Desc *result, /* result descriptor */
                            F90_Desc *array,  /* array descriptor */
                            F90_Desc *mask,   /* mask descriptor */
                            F90_Desc *vector  /* vector descriptor */
                            DCLEN64(rb)         /* result char len */
                            DCLEN64(ab))        /* array char len */
{
  ENTFTN(PACKZ, packz)(CADR(rb), CADR(ab), mb, result, array, mask);
}
/* 32 bit CLEN version */
void ENTFTN(PACKZC, packzc)(DCHAR(rb),        /* result char base */
                            DCHAR(ab),        /* array char base */
                            void *mb,         /* mask base */
                            F90_Desc *result, /* result descriptor */
                            F90_Desc *array,  /* array descriptor */
                            F90_Desc *mask,   /* mask descriptor */
                            F90_Desc *vector  /* vector descriptor */
                            DCLEN(rb)         /* result char len */
                            DCLEN(ab))        /* array char len */
{
  ENTFTN(PACKZCA, packzca)(CADR(rb), CADR(ab), mb, result, array, mask,
                           vector, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab));
}

/* unpack */

void ENTFTN(UNPACK, unpack)(void *rb,         /* result base */
                            void *vb,         /* vector base */
                            void *mb,         /* mask base */
                            void *fb,         /* field base */
                            F90_Desc *result, /* result descriptor */
                            F90_Desc *vector, /* vector descriptor */
                            F90_Desc *mask,   /* mask descriptor */
                            F90_Desc *field)  /* field descriptor */
{
  char *la, *rf;
  __INT_T rindex[MAXDIMS];
  __INT_T vindex;
  __INT_T findex[MAXDIMS];
  __INT_T mindex[MAXDIMS];
  __BIGREAL_T tmp[4];
  __INT_T field_is_array, i, more, mval;

  if (result == NULL || F90_TAG_G(result) != __DESC)
    __fort_abort("UNPACK: invalid result descriptor");

  if (F90_GSIZE_G(result) == 0 || F90_GSIZE_G(mask) == 0)
    return;

  rf = (char *)rb + DIST_SCOFF_G(result) * F90_LEN_G(result);

  for (i = F90_RANK_G(result); --i >= 0;)
    rindex[i] = F90_DIM_LBOUND_G(result, i);

  if (mask == NULL || F90_TAG_G(mask) != __DESC)
    __fort_abort("UNPACK: invalid mask descriptor");

  for (i = F90_RANK_G(mask); --i >= 0;)
    mindex[i] = F90_DIM_LBOUND_G(mask, i);

  vindex = F90_DIM_LBOUND_G(vector, 0);

  if (ISSCALAR(field)) {
    field_is_array = 0;
  } else if (F90_TAG_G(field) == __DESC) {
    for (i = F90_RANK_G(field); --i >= 0;)
      findex[i] = F90_DIM_LBOUND_G(field, i);
    field_is_array = 1;
  } else
    __fort_abort("UNPACK: invalid field descriptor");

  more = 1;
  while (more) {

    /* get mask value */

    I8(__fort_get_scalar)(tmp, mb, mask, mindex);
    switch (F90_KIND_G(mask)) {
    case __LOG1:
      mval = (*(__LOG1_T *)tmp & GET_DIST_MASK_LOG1) != 0;
      break;
    case __LOG2:
      mval = (*(__LOG2_T *)tmp & GET_DIST_MASK_LOG2) != 0;
      break;
    case __LOG4:
      mval = (*(__LOG4_T *)tmp & GET_DIST_MASK_LOG4) != 0;
      break;
    case __LOG8:
      mval = (*(__LOG8_T *)tmp & GET_DIST_MASK_LOG8) != 0;
      break;
    case __INT1:
      mval = (*(__INT1_T *)tmp & GET_DIST_MASK_INT1) != 0;
      break;
    case __INT2:
      mval = (*(__INT2_T *)tmp & GET_DIST_MASK_INT2) != 0;
      break;
    case __INT4:
      mval = (*(__INT4_T *)tmp & GET_DIST_MASK_INT4) != 0;
      break;
    case __INT8:
      mval = (*(__INT8_T *)tmp & GET_DIST_MASK_INT8) != 0;
      break;
    default:
      __fort_abort("UNPACK: unknown mask type");
    }

    /* if the mask is true, move the next vector element to the
       result element corresponding to the mask.  Otherwise, copy
       the field element corresponding to the mask to the result
       element. */

    la = I8(__fort_local_address)(rf, result, rindex);
    if (la == NULL)
      la = (char *)tmp;
    if (mval) {
      I8(__fort_get_scalar)(la, vb, vector, &vindex);
      I8(next_index)(&vindex, vector);
    } else if (field_is_array)
      I8(__fort_get_scalar)(la, fb, field, findex);
    else
      __fort_scalar_copy[F90_KIND_G(result)](la, fb, F90_LEN_G(result));

    more &= I8(next_index)(rindex, result);
    more &= I8(next_index)(mindex, mask);
    if (field_is_array)
      more &= I8(next_index)(findex, field);
  }
}

void ENTFTN(UNPACKCA, unpackca)(DCHAR(rb),        /* result char base */
                              DCHAR(vb),        /* vector char base */
                              void *mb,         /* mask base */
                              DCHAR(fb),        /* field char base */
                              F90_Desc *result, /* result descriptor */
                              F90_Desc *vector, /* vector descriptor */
                              F90_Desc *mask,   /* mask descriptor */
                              F90_Desc *field   /* field descriptor */
                              DCLEN64(rb)         /* result char len */
                              DCLEN64(vb)         /* vector char len */
                              DCLEN64(fb))        /* field char len */
{
  ENTFTN(UNPACK,unpack)(CADR(rb), CADR(vb), mb, CADR(fb),
			  result, vector, mask, field);
}
/* 32 bit CLEN version */
void ENTFTN(UNPACKC, unpackc)(DCHAR(rb),        /* result char base */
                              DCHAR(vb),        /* vector char base */
                              void *mb,         /* mask base */
                              DCHAR(fb),        /* field char base */
                              F90_Desc *result, /* result descriptor */
                              F90_Desc *vector, /* vector descriptor */
                              F90_Desc *mask,   /* mask descriptor */
                              F90_Desc *field   /* field descriptor */
                              DCLEN(rb)         /* result char len */
                              DCLEN(vb)         /* vector char len */
                              DCLEN(fb))        /* field char len */
{
  ENTFTN(UNPACKCA, unpackca)(CADR(rb), CADR(vb), mb, CADR(fb), result, vector,
      mask, field, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(vb), (__CLEN_T)CLEN(fb));
}
