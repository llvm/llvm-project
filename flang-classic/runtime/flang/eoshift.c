/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "fort_vars.h"

static void I8(eoshift_scalar)(char *rb,          /* result base */
                               char *ab,          /* array base */
                               __INT_T shift_amt, /* shift amount */
                               const char *bb,    /* boundary base */
                               __INT_T shift_dim, /* shift dimension */
                               F90_Desc *rs,      /* result descriptor */
                               F90_Desc *as,      /* array descriptor */
                               F90_Desc *rc, /* result subsection descriptor */
                               F90_Desc *ac, /* array subsection descriptor */
                               __INT_T sub_dim) /* subsection dimension */
{
  chdr *c;
  char *ap, *rp;
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(rsd);
  __INT_T aflags, albase, apbase, arepli, ascoff;
  __INT_T rflags, rlbase, rpbase, rrepli, rscoff;
  __INT_T aolb[MAXDIMS], aoub[MAXDIMS];
  __INT_T rolb[MAXDIMS], roub[MAXDIMS];
  __INT_T extent, i, sabs;

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d eoshift_scalar shift=%d boundary=", GET_DIST_LCPU, shift_amt);
    __fort_print_scalar(bb, F90_KIND_G(rs));
    printf(" dim=%d\n", shift_dim);
  }
#endif

  SET_DIM_PTRS(rsd, rs, shift_dim - 1);
  SET_DIM_PTRS(asd, as, shift_dim - 1);

  extent = F90_DPTR_EXTENT_G(asd);
  if (extent < 0)
    return;

  /* save descriptor fields affected by set/finish_section */

  aflags = F90_FLAGS_G(ac);
  albase = F90_LBASE_G(ac);
  apbase = DIST_PBASE_G(ac);
  arepli = DIST_REPLICATED_G(ac);
  ascoff = DIST_SCOFF_G(ac);
  for (i = F90_RANK_G(ac); --i >= 0;) {
    aolb[i] = DIST_DIM_OLB_G(ac, i);
    aoub[i] = DIST_DIM_OUB_G(ac, i);
  }
  rflags = F90_FLAGS_G(rc);
  rlbase = F90_LBASE_G(rc);
  rpbase = DIST_PBASE_G(rc);
  rrepli = DIST_REPLICATED_G(rc);
  rscoff = DIST_SCOFF_G(rc);
  for (i = F90_RANK_G(rc); --i >= 0;) {
    rolb[i] = DIST_DIM_OLB_G(rc, i);
    roub[i] = DIST_DIM_OUB_G(rc, i);
  }

  /* copy directly if shift amount is zero */

  if (shift_amt == 0) {
    I8(__fort_set_section)(rc, sub_dim, rs, shift_dim,
			      F90_DPTR_LBOUND_G(rsd), DPTR_UBOUND_G(rsd), 1);
    I8(__fort_finish_section)(rc);
    I8(__fort_set_section)(ac, sub_dim, as, shift_dim,
			      F90_DPTR_LBOUND_G(asd), DPTR_UBOUND_G(asd), 1);
    I8(__fort_finish_section)(ac);

    /* adjust base addresses for scalar subscripts and copy */

    rp = rb + DIST_SCOFF_G(rc) * F90_LEN_G(rc);
    ap = ab + DIST_SCOFF_G(ac) * F90_LEN_G(ac);
    c = I8(__fort_copy)(rp, ap, rc, ac, NULL);
    __fort_doit(c);
    __fort_frechn(c);

    /* restore descriptor fields */

    F90_FLAGS_P(ac, aflags);
    F90_LBASE_P(ac, albase);
    DIST_PBASE_P(ac, apbase);
    DIST_REPLICATED_P(ac, arepli);
    DIST_SCOFF_P(ac, ascoff);
    for (i = F90_RANK_G(ac); --i >= 0;) {
      DIST_DIM_OLB_P(ac, i, aolb[i]);
      DIST_DIM_OUB_P(ac, i, aoub[i]);
    }
    DIST_CACHED_P(ac, 0);

    F90_FLAGS_P(rc, rflags);
    F90_LBASE_P(rc, rlbase);
    DIST_PBASE_P(rc, rpbase);
    DIST_REPLICATED_P(rc, rrepli);
    DIST_SCOFF_P(rc, rscoff);
    for (i = F90_RANK_G(rc); --i >= 0;) {
      DIST_DIM_OLB_P(rc, i, rolb[i]);
      DIST_DIM_OUB_P(rc, i, roub[i]);
    }
    DIST_CACHED_P(rc, 0);
    return;
  }

  /* if the absolute shift amount is greater than or equal to the
     extent, just fill the result section with boundary values */

  sabs = Abs(shift_amt);
  if (sabs >= extent) {
    I8(__fort_set_section)(rc, sub_dim, rs, shift_dim,
			      F90_DPTR_LBOUND_G(rsd), DPTR_UBOUND_G(rsd), 1);
    I8(__fort_finish_section)(rc);

    I8(__fort_fills)(rb, rc, bb);

    /* restore descriptor fields */

    F90_FLAGS_P(rc, rflags);
    F90_LBASE_P(rc, rlbase);
    DIST_PBASE_P(rc, rpbase);
    DIST_REPLICATED_P(rc, rrepli);
    DIST_SCOFF_P(rc, rscoff);
    DIST_CACHED_P(rc, 0);
    for (i = F90_RANK_G(rc); --i >= 0;) {
      DIST_DIM_OLB_P(rc, i, rolb[i]);
      DIST_DIM_OUB_P(rc, i, roub[i]);
    }
    return;
  }

  if (shift_amt < 0)
    sabs = extent - sabs;

  /* lower part of result */

  I8(__fort_set_section)(rc, sub_dim, rs, shift_dim,
		          F90_DPTR_LBOUND_G(rsd), DPTR_UBOUND_G(rsd) - sabs, 1);

  I8(__fort_finish_section)(rc);

  if (shift_amt > 0) {

    I8(__fort_set_section)(ac, sub_dim, as, shift_dim,
			      F90_DPTR_LBOUND_G(asd) + sabs, 
                              DPTR_UBOUND_G(asd), 1);

    I8(__fort_finish_section)(ac);
    rp = rb + DIST_SCOFF_G(rc) * F90_LEN_G(rc);
    ap = ab + DIST_SCOFF_G(ac) * F90_LEN_G(ac);
    c = I8(__fort_copy)(rp, ap, rc, ac, NULL);
  } else
    I8(__fort_fills)(rb, rc, bb);

  /* restore descriptor fields */

  F90_FLAGS_P(ac, aflags);
  F90_LBASE_P(ac, albase);
  DIST_PBASE_P(ac, apbase);
  DIST_REPLICATED_P(ac, arepli);
  DIST_SCOFF_P(ac, ascoff);
  DIST_CACHED_P(ac, 0);
  for (i = F90_RANK_G(ac); --i >= 0;) {
    DIST_DIM_OLB_P(ac, i, aolb[i]);
    DIST_DIM_OUB_P(ac, i, aoub[i]);
  }
  F90_FLAGS_P(rc, rflags);
  F90_LBASE_P(rc, rlbase);
  DIST_PBASE_P(rc, rpbase);
  DIST_REPLICATED_P(rc, rrepli);
  DIST_SCOFF_P(rc, rscoff);
  for (i = F90_RANK_G(rc); --i >= 0;) {
    DIST_DIM_OLB_P(rc, i, rolb[i]);
    DIST_DIM_OUB_P(rc, i, roub[i]);
  }
  DIST_CACHED_P(rc, 0);

  /* upper part of result */

  I8(__fort_set_section)(rc, sub_dim, rs, shift_dim,
		          F90_DPTR_LBOUND_G(rsd) + (extent - sabs), 
                          DPTR_UBOUND_G(rsd), 1);

  I8(__fort_finish_section)(rc);

  if (shift_amt < 0) {

    I8(__fort_set_section)(ac, sub_dim, as, shift_dim,
			      F90_DPTR_LBOUND_G(asd), 
                              DPTR_UBOUND_G(asd) - (extent - sabs), 1);

    I8(__fort_finish_section)(ac);
    rp = rb + DIST_SCOFF_G(rc) * F90_LEN_G(rc);
    ap = ab + DIST_SCOFF_G(ac) * F90_LEN_G(ac);
    c = I8(__fort_copy)(rp, ap, rc, ac, NULL);
  } else
    I8(__fort_fills)(rb, rc, bb);

  __fort_doit(c);
  __fort_frechn(c);

  /* restore descriptor fields */

  F90_FLAGS_P(ac, aflags);
  F90_LBASE_P(ac, albase);
  DIST_PBASE_P(ac, apbase);
  DIST_REPLICATED_P(ac, arepli);
  DIST_SCOFF_P(ac, ascoff);
  for (i = F90_RANK_G(ac); --i >= 0;) {
    DIST_DIM_OLB_P(ac, i, aolb[i]);
    DIST_DIM_OUB_P(ac, i, aoub[i]);
  }
  DIST_CACHED_P(ac, 0);

  F90_FLAGS_P(rc, rflags);
  F90_LBASE_P(rc, rlbase);
  DIST_PBASE_P(rc, rpbase);
  DIST_REPLICATED_P(rc, rrepli);
  DIST_SCOFF_P(rc, rscoff);
  for (i = F90_RANK_G(rc); --i >= 0;) {
    DIST_DIM_OLB_P(rc, i, rolb[i]);
    DIST_DIM_OUB_P(rc, i, roub[i]);
  }
  DIST_CACHED_P(rc, 0);
}

static void I8(eoshift_loop)(char *rb,          /* result base */
                             char *ab,          /* array base */
                             __INT_T *sb,       /* shift base */
                             const char *bb,    /* boundary base */
                             __INT_T shift_dim, /* dimension to shift */
                             F90_Desc *rs,      /* result descriptor */
                             F90_Desc *as,      /* array descriptor */
                             F90_Desc *ss,      /* shift descriptor */
                             F90_Desc *bs,      /* boundary descriptor */
                             F90_Desc *rc, /* result subsection descriptor */
                             F90_Desc *ac, /* array subsection descriptor */
                             __INT_T soff, /* shift offset */
                             __INT_T boff, /* boundary offset */
                             __INT_T loop_dim) /* loop dim */
{
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(bsd);
  DECL_DIM_PTRS(rsd);
  DECL_DIM_PTRS(ssd);
  __INT_T aflags, albase, apbase, arepli, ascoff;
  __INT_T rflags, rlbase, rpbase, rrepli, rscoff;
  __INT_T ai, array_dim, bstr, ri, sstr;

  /* shift rank = array rank - 1*/

  array_dim = loop_dim;
  if (array_dim >= shift_dim)
    ++array_dim;

  SET_DIM_PTRS(rsd, rs, array_dim - 1);
  ri = F90_DPTR_LBOUND_G(rsd);

  SET_DIM_PTRS(asd, as, array_dim - 1);
  ai = F90_DPTR_LBOUND_G(asd);

  if (F90_TAG_G(ss) == __DESC) {
    SET_DIM_PTRS(ssd, ss, loop_dim - 1);
    sstr = F90_DPTR_SSTRIDE_G(ssd) * F90_DPTR_LSTRIDE_G(ssd);
    soff += (F90_DPTR_SSTRIDE_G(ssd) * F90_DPTR_LBOUND_G(ssd) +
             F90_DPTR_SOFFSET_G(ssd)) *
            F90_DPTR_LSTRIDE_G(ssd);
  } else
    sstr = soff = 0;

  if (F90_TAG_G(bs) == __DESC) {
    SET_DIM_PTRS(bsd, bs, loop_dim - 1);
    bstr = F90_DPTR_SSTRIDE_G(bsd) * F90_DPTR_LSTRIDE_G(bsd);
    boff += (F90_DPTR_SSTRIDE_G(bsd) * F90_DPTR_LBOUND_G(bsd) +
             F90_DPTR_SOFFSET_G(bsd)) *
            F90_DPTR_LSTRIDE_G(bsd);
  } else
    bstr = boff = 0;

  /* save descriptor fields affected by set_single */

  aflags = F90_FLAGS_G(ac);
  albase = F90_LBASE_G(ac);
  apbase = DIST_PBASE_G(ac);
  arepli = DIST_REPLICATED_G(ac);
  ascoff = DIST_SCOFF_G(ac);

  rflags = F90_FLAGS_G(rc);
  rlbase = F90_LBASE_G(rc);
  rpbase = DIST_PBASE_G(rc);
  rrepli = DIST_REPLICATED_G(rc);
  rscoff = DIST_SCOFF_G(rc);

  for (; ri <= DPTR_UBOUND_G(rsd); ++ri, ++ai, soff += sstr, boff += bstr) {
    I8(__fort_set_single)(rc, rs, array_dim, ri, __SCALAR);
    I8(__fort_set_single)(ac, as, array_dim, ai, __SCALAR);

    if (loop_dim > 1)
      I8(eoshift_loop)(rb, ab, sb, bb, shift_dim, rs, as, ss, bs, rc, ac,
			     soff, boff, loop_dim-1);
    else

      I8(eoshift_scalar)(rb, ab, sb[soff], bb + boff*F90_LEN_G(bs),
			       shift_dim, rs, as, rc, ac, 1);

    /* restore descriptor fields */

    F90_FLAGS_P(ac, aflags);
    F90_LBASE_P(ac, albase);
    DIST_PBASE_P(ac, apbase);
    DIST_REPLICATED_P(ac, arepli);
    DIST_SCOFF_P(ac, ascoff);
    DIST_CACHED_P(ac, 0);

    F90_FLAGS_P(rc, rflags);
    F90_LBASE_P(rc, rlbase);
    DIST_PBASE_P(rc, rpbase);
    DIST_REPLICATED_P(rc, rrepli);
    DIST_SCOFF_P(rc, rscoff);
    DIST_CACHED_P(rc, 0);
  }
}

/* eoshift (..., shift=scalar), boundary absent */

void ENTFTN(EOSHIFTSZ, eoshiftsz)(char *rb,     /* result base */
                                  char *ab,     /* array base */
                                  __INT_T *sb,  /* shift base */
                                  __INT_T *db,  /* dimension */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds) /* dim descriptor */
{
  const char *bb;
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(rsd);
  __INT_T dim, i, shift;

  shift = *sb;
  dim = *db;
  bb = (F90_KIND_G(rs) == __STR) ? " " : (const char *)GET_DIST_ZED;

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = EOSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=%d, dim=%d)\n", ab, shift, dim);
  }
#endif

  /* initialize section descriptors */

  __DIST_INIT_SECTION(ac, F90_RANK_G(as), as);
  __DIST_INIT_SECTION(rc, F90_RANK_G(rs), rs);

  for (i = 1; i <= F90_RANK_G(as); ++i) {
    if (i == dim)
      continue;
    SET_DIM_PTRS(asd, as, i - 1);
    I8(__fort_set_section)(ac, i, as, i, F90_DPTR_LBOUND_G(asd), 
                              DPTR_UBOUND_G(asd), 1);
    SET_DIM_PTRS(rsd, rs, i - 1);
    I8(__fort_set_section)(rc, i, rs, i, F90_DPTR_LBOUND_G(rsd), 
                              DPTR_UBOUND_G(rsd), 1);
  }

  I8(eoshift_scalar)(rb, ab, shift, bb, dim, rs, as, rc, ac, dim);
}

void ENTFTN(EOSHIFTSZCA, eoshiftszca)(DCHAR(rb),    /* result char base */
                                    DCHAR(ab),    /* array char base */
                                    __INT_T *sb,  /* shift base */
                                    __INT_T *db,  /* dimension */
                                    F90_Desc *rs, /* result descriptor */
                                    F90_Desc *as, /* array descriptor */
                                    F90_Desc *ss, /* shift descriptor */
                                    F90_Desc *ds  /* dim descriptor */
                                    DCLEN64(rb)     /* result char len */
                                    DCLEN64(ab))    /* array char len */
{
  ENTFTN(EOSHIFTSZ, eoshiftsz)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds);
}

/* 32 bit CLEN version */
void ENTFTN(EOSHIFTSZC, eoshiftszc)(DCHAR(rb),    /* result char base */
                                    DCHAR(ab),    /* array char base */
                                    __INT_T *sb,  /* shift base */
                                    __INT_T *db,  /* dimension */
                                    F90_Desc *rs, /* result descriptor */
                                    F90_Desc *as, /* array descriptor */
                                    F90_Desc *ss, /* shift descriptor */
                                    F90_Desc *ds  /* dim descriptor */
                                    DCLEN(rb)     /* result char len */
                                    DCLEN(ab))    /* array char len */
{
  ENTFTN(EOSHIFTSZCA, eoshiftszca)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds,
                                   (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab));
}

/* eoshift (..., shift=scalar, boundary=scalar) */

void ENTFTN(EOSHIFTSS, eoshiftss)(char *rb,     /* result base */
                                  char *ab,     /* array base */
                                  __INT_T *sb,  /* shift base */
                                  __INT_T *db,  /* dimension */
                                  char *bb,     /* boundary base */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds, /* dim descriptor */
                                  F90_Desc *bs) /* boundary descriptor */
{
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  DECL_DIM_PTRS(asd);
  DECL_DIM_PTRS(rsd);
  __INT_T dim, i, shift;

  shift = *sb;
  dim = *db;

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = EOSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=%d, boundary=", ab, shift);
    __fort_print_scalar(bb, (dtype)F90_TAG_G(bs));
    printf(", dim=%d)\n", dim);
  }
#endif

  /* initialize section descriptors */

  __DIST_INIT_SECTION(ac, F90_RANK_G(as), as);
  __DIST_INIT_SECTION(rc, F90_RANK_G(rs), rs);

  for (i = 1; i <= F90_RANK_G(as); ++i) {
    if (i == dim)
      continue;
    SET_DIM_PTRS(asd, as, i - 1);
    I8(__fort_set_section)(ac, i, as, i, F90_DPTR_LBOUND_G(asd), 
                              DPTR_UBOUND_G(asd), 1);
    SET_DIM_PTRS(rsd, rs, i - 1);
    I8(__fort_set_section)(rc, i, rs, i, F90_DPTR_LBOUND_G(rsd), 
                              DPTR_UBOUND_G(rsd), 1);
  }

  I8(eoshift_scalar)(rb, ab, shift, bb, dim, rs, as, rc, ac, dim);
}

void ENTFTN(EOSHIFTSSCA, eoshiftssca)(DCHAR(rb),    /* result char base */
                                    DCHAR(ab),    /* array char base */
                                    __INT_T *sb,  /* shift base */
                                    __INT_T *db,  /* dimension */
                                    DCHAR(bb),    /* boundary char base */
                                    F90_Desc *rs, /* result descriptor */
                                    F90_Desc *as, /* array descriptor */
                                    F90_Desc *ss, /* shift descriptor */
                                    F90_Desc *ds, /* dim descriptor */
                                    F90_Desc *bs  /* boundary descriptor */
                                    DCLEN64(rb)     /* result char len */
                                    DCLEN64(ab)     /* array char len */
                                    DCLEN64(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTSS,eoshiftss)(CADR(rb), CADR(ab), sb, db, CADR(bb),
				rs, as, ss, ds, bs);
}

/* 32 bit CLEN version */
void ENTFTN(EOSHIFTSSC, eoshiftssc)(DCHAR(rb),    /* result char base */
                                    DCHAR(ab),    /* array char base */
                                    __INT_T *sb,  /* shift base */
                                    __INT_T *db,  /* dimension */
                                    DCHAR(bb),    /* boundary char base */
                                    F90_Desc *rs, /* result descriptor */
                                    F90_Desc *as, /* array descriptor */
                                    F90_Desc *ss, /* shift descriptor */
                                    F90_Desc *ds, /* dim descriptor */
                                    F90_Desc *bs  /* boundary descriptor */
                                    DCLEN(rb)     /* result char len */
                                    DCLEN(ab)     /* array char len */
                                    DCLEN(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTSSCA, eoshiftssca)(CADR(rb), CADR(ab), sb, db, CADR(bb), rs, as, 
                                   ss, ds, bs, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab), 
                                   (__CLEN_T)CLEN(bb));
}

/* eoshift (..., shift=scalar, boundary=array) */

void ENTFTN(EOSHIFTSA, eoshiftsa)(char *rb,     /* result base */
                                  char *ab,     /* array base */
                                  __INT_T *sb,  /* shift base */
                                  __INT_T *db,  /* dimension */
                                  char *bb,     /* boundary base */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds, /* dim descriptor */
                                  F90_Desc *bs) /* boundary descriptor */
{
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  __INT_T dim, shift;

  shift = *sb;
  dim = *db;

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = EOSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=%d, boundary=b", ab, shift);
    I8(__fort_show_section)(bs);
    printf("@%x, dim=%d)\n", bb, dim);
  }
#endif

  /* initialize rank 1 section descriptors */

  __DIST_INIT_SECTION(rc, 1, rs);
  __DIST_INIT_SECTION(ac, 1, as);

  I8(eoshift_loop)(rb, ab, sb, bb, dim, rs, as, ss, bs,
		     rc, ac, 0, F90_LBASE_G(bs) - 1, F90_RANK_G(bs));
}

void ENTFTN(EOSHIFTSACA, eoshiftsaca)(DCHAR(rb),    /* result char base */
                                    DCHAR(ab),    /* array char base */
                                    __INT_T *sb,  /* shift char base */
                                    __INT_T *db,  /* dimension */
                                    DCHAR(bb),    /* boundary base */
                                    F90_Desc *rs, /* result descriptor */
                                    F90_Desc *as, /* array descriptor */
                                    F90_Desc *ss, /* shift descriptor */
                                    F90_Desc *ds, /* dim descriptor */
                                    F90_Desc *bs  /* boundary descriptor */
                                    DCLEN64(rb)     /* result char len */
                                    DCLEN64(ab)     /* array char len */
                                    DCLEN64(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTSA,eoshiftsa)(CADR(rb), CADR(ab), sb, db, CADR(bb),
				rs, as, ss, ds, bs);
}

/* 32 bit CLEN version */
void ENTFTN(EOSHIFTSAC, eoshiftsac)(DCHAR(rb),    /* result char base */
                                    DCHAR(ab),    /* array char base */
                                    __INT_T *sb,  /* shift char base */
                                    __INT_T *db,  /* dimension */
                                    DCHAR(bb),    /* boundary base */
                                    F90_Desc *rs, /* result descriptor */
                                    F90_Desc *as, /* array descriptor */
                                    F90_Desc *ss, /* shift descriptor */
                                    F90_Desc *ds, /* dim descriptor */
                                    F90_Desc *bs  /* boundary descriptor */
                                    DCLEN(rb)     /* result char len */
                                    DCLEN(ab)     /* array char len */
                                    DCLEN(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTSACA, eoshiftsaca)(CADR(rb), CADR(ab), sb, db, CADR(bb), rs, as,
                                   ss, ds, bs, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab),
                                   (__CLEN_T)CLEN(bb));
}

/* eoshift (..., shift=array), boundary absent */

void ENTFTN(EOSHIFTZ, eoshiftz)(char *rb,     /* result base */
                                char *ab,     /* array base */
                                __INT_T *sb,  /* shift base */
                                __INT_T *db,  /* dimension to shift */
                                F90_Desc *rs, /* result descriptor */
                                F90_Desc *as, /* array descriptor */
                                F90_Desc *ss, /* shift descriptor */
                                F90_Desc *ds) /* dim descriptor */
{
  DECL_HDR_PTRS(bs);
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  const char *bb;
  __INT_T dim;

  dim = *db;
  bb = (F90_KIND_G(rs) == __STR) ? " " : (const char *)GET_DIST_ZED;
  bs = (F90_Desc *)&F90_KIND_G(rs);

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = EOSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=s", ab);
    I8(__fort_show_section)(ss);
    printf("@%x, dim=%d)\n", sb, dim);
  }
#endif

  /* initialize rank 1 section descriptors */

  __DIST_INIT_SECTION(rc, 1, rs);
  __DIST_INIT_SECTION(ac, 1, as);

  /* loop over all shift array elements */

  I8(eoshift_loop)(rb, ab, sb, bb, dim, rs, as, ss, bs,
		     rc, ac, F90_LBASE_G(ss) - 1, 0, F90_RANK_G(ss));
}

void ENTFTN(EOSHIFTZCA, eoshiftzca)(DCHAR(rb),    /* result char base */
                                  DCHAR(ab),    /* array char base */
                                  __INT_T *sb,  /* shift base */
                                  __INT_T *db,  /* dimension to shift */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds  /* dim descriptor */
                                  DCLEN64(rb)     /* result char len */
                                  DCLEN64(ab))    /* array char len */
{
  ENTFTN(EOSHIFTZ, eoshiftz)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds);
}

/* 32 bit CLEN version */
void ENTFTN(EOSHIFTZC, eoshiftzc)(DCHAR(rb),    /* result char base */
                                  DCHAR(ab),    /* array char base */
                                  __INT_T *sb,  /* shift base */
                                  __INT_T *db,  /* dimension to shift */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds  /* dim descriptor */
                                  DCLEN(rb)     /* result char len */
                                  DCLEN(ab))    /* array char len */
{
  ENTFTN(EOSHIFTZCA, eoshiftzca)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds,
                                  (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab));
}

/* eoshift (..., shift=array, boundary=scalar) */

void ENTFTN(EOSHIFTS, eoshifts)(char *rb,     /* result base */
                                char *ab,     /* array base */
                                __INT_T *sb,  /* shift base */
                                __INT_T *db,  /* dimension to shift */
                                char *bb,     /* boundary base */
                                F90_Desc *rs, /* result descriptor */
                                F90_Desc *as, /* array descriptor */
                                F90_Desc *ss, /* shift descriptor */
                                F90_Desc *ds, /* dim descriptor */
                                F90_Desc *bs) /* boundary descriptor */
{
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  __INT_T dim;

  dim = *db;

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = EOSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=s", ab);
    I8(__fort_show_section)(ss);
    printf("@%x, boundary=", sb);
    __fort_print_scalar(bb, (dtype)F90_TAG_G(bs));
    printf(", dim=%d)\n", dim);
  }
#endif

  /* initialize rank 1 section descriptors */

  __DIST_INIT_SECTION(rc, 1, rs);
  __DIST_INIT_SECTION(ac, 1, as);

  /* loop over all shift array elements */

  I8(eoshift_loop)(rb, ab, sb, bb, dim, rs, as, ss, bs,
		     rc, ac, F90_LBASE_G(ss) - 1, 0, F90_RANK_G(ss));
}

void ENTFTN(EOSHIFTSCA, eoshiftsca)(DCHAR(rb),    /* result char base */
                                  DCHAR(ab),    /* array char base */
                                  __INT_T *sb,  /* shift char base */
                                  __INT_T *db,  /* dimension to shift */
                                  DCHAR(bb),    /* boundary base */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds, /* dim descriptor */
                                  F90_Desc *bs  /* boundary descriptor */
                                  DCLEN64(rb)     /* result char len */
                                  DCLEN64(ab)     /* array char len */
                                  DCLEN64(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTS,eoshifts)(CADR(rb), CADR(ab), sb, db, CADR(bb),
			      rs, as, ss, ds, bs);
}

/* 32 bit CLEN version */
void ENTFTN(EOSHIFTSC, eoshiftsc)(DCHAR(rb),    /* result char base */
                                  DCHAR(ab),    /* array char base */
                                  __INT_T *sb,  /* shift char base */
                                  __INT_T *db,  /* dimension to shift */
                                  DCHAR(bb),    /* boundary base */
                                  F90_Desc *rs, /* result descriptor */
                                  F90_Desc *as, /* array descriptor */
                                  F90_Desc *ss, /* shift descriptor */
                                  F90_Desc *ds, /* dim descriptor */
                                  F90_Desc *bs  /* boundary descriptor */
                                  DCLEN(rb)     /* result char len */
                                  DCLEN(ab)     /* array char len */
                                  DCLEN(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTSCA, eoshiftsca)(CADR(rb), CADR(ab), sb, db, CADR(bb), rs, as,
                                  ss, ds, bs, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab),
                                  (__CLEN_T)CLEN(bb));
}

/* eoshift (..., shift=array, boundary=array) */

void ENTFTN(EOSHIFT, eoshift)(char *rb,     /* result base */
                              char *ab,     /* array base */
                              __INT_T *sb,  /* shift base */
                              __INT_T *db,  /* dimension to shift */
                              char *bb,     /* boundary base */
                              F90_Desc *rs, /* result descriptor */
                              F90_Desc *as, /* array descriptor */
                              F90_Desc *ss, /* shift descriptor */
                              F90_Desc *ds, /* dim descriptor */
                              F90_Desc *bs) /* boundary descriptor */
{
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  __INT_T dim;

  dim = *db;

#if defined(DEBUG)
  if (__fort_test & DEBUG_EOSH) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = EOSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=s", ab);
    I8(__fort_show_section)(ss);
    printf("@%x, boundary=b", sb);
    I8(__fort_show_section)(bs);
    printf("@%x, dim=%d)\n", bb, dim);
  }
#endif

  /* initialize rank 1 section descriptors */

  __DIST_INIT_SECTION(rc, 1, rs);
  __DIST_INIT_SECTION(ac, 1, as);

  /* loop over all shift array elements */

  I8(eoshift_loop)(rb, ab, sb, bb, dim, rs, as, ss, bs,
		     rc, ac, F90_LBASE_G(ss) - 1, F90_LBASE_G(bs) - 1, 
                     F90_RANK_G(ss));
}

void ENTFTN(EOSHIFTCA, eoshiftca)(DCHAR(rb),    /* result base */
                                DCHAR(ab),    /* array base */
                                __INT_T *sb,  /* shift base */
                                __INT_T *db,  /* dimension to shift */
                                DCHAR(bb),    /* boundary base */
                                F90_Desc *rs, /* result descriptor */
                                F90_Desc *as, /* array descriptor */
                                F90_Desc *ss, /* shift descriptor */
                                F90_Desc *ds, /* dim descriptor */
                                F90_Desc *bs  /* boundary descriptor */
                                DCLEN64(rb)     /* result char len */
                                DCLEN64(ab)     /* array char len */
                                DCLEN64(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFT,eoshift)(CADR(rb), CADR(ab), sb, db, CADR(bb),
			    rs, as, ss, ds, bs);
}

/* 32 bit CLEN version */
void ENTFTN(EOSHIFTC, eoshiftc)(DCHAR(rb),    /* result base */
                                DCHAR(ab),    /* array base */
                                __INT_T *sb,  /* shift base */
                                __INT_T *db,  /* dimension to shift */
                                DCHAR(bb),    /* boundary base */
                                F90_Desc *rs, /* result descriptor */
                                F90_Desc *as, /* array descriptor */
                                F90_Desc *ss, /* shift descriptor */
                                F90_Desc *ds, /* dim descriptor */
                                F90_Desc *bs  /* boundary descriptor */
                                DCLEN(rb)     /* result char len */
                                DCLEN(ab)     /* array char len */
                                DCLEN(bb))    /* boundary char len */
{
  ENTFTN(EOSHIFTCA, eoshiftca)(CADR(rb), CADR(ab), sb, db, CADR(bb), rs, as, ss,
                                ds, bs, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab),
                                (__CLEN_T)CLEN(bb));
}
