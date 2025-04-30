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

/* result = cshift(array, shift=scalar, dim) */

void ENTFTN(CSHIFTS, cshifts)(void *rb,     /* result base */
                              void *ab,     /* array base */
                              __INT_T *sb,  /* shift base */
                              __INT_T *db,  /* dimension */
                              F90_Desc *rs, /* result descriptor */
                              F90_Desc *as, /* array descriptor */
                              F90_Desc *ss, /* shift descriptor */
                              F90_Desc *ds) /* dim descriptor */
{
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(rd);
  char *ap, *rp;
  chdr *c1, *c2;
  __INT_T aflags, albase, apbase, arepli, ascoff;
  __INT_T rflags, rlbase, rpbase, rrepli, rscoff;
  __INT_T aolb[MAXDIMS], aoub[MAXDIMS];
  __INT_T rolb[MAXDIMS], roub[MAXDIMS];
  __INT_T dim, extent, i, sabs, shift;

  shift = *sb;
  dim = *db;

#if defined(DEBUG)
  if (__fort_test & DEBUG_CSHF) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = CSHIFT(a", rb);
    I8(__fort_show_section)(as);
    printf("@%x, shift=%d, dim=%d)\n", ab, shift, dim);
  }
#endif

  /* compute the net positive (left) shift amount */

  SET_DIM_PTRS(ad, as, dim - 1);
  extent = F90_DPTR_EXTENT_G(ad);
  if (extent < 0)
    return;

  sabs = shift % extent;
  if (sabs < 0)
    sabs += extent;

  /* copy straight across if net shift amount is zero */

  if (sabs == 0) {
    rp = (char *)rb + DIST_SCOFF_G(rs) * F90_LEN_G(rs);
    ap = (char *)ab + DIST_SCOFF_G(as) * F90_LEN_G(as);
    c1 = I8(__fort_copy)(rp, ap, rs, as, NULL);
    __fort_doit(c1);
    __fort_frechn(c1);
    return;
  }

  /* form section descriptors to describe shift. */

  __DIST_INIT_SECTION(ac, F90_RANK_G(as), as);
  __DIST_INIT_SECTION(rc, F90_RANK_G(as), rs);

  for (i = 1; i <= F90_RANK_G(as); ++i) {
    if (i == dim)
      continue;
    SET_DIM_PTRS(ad, as, i - 1);
    I8(__fort_set_section)(ac, i, as, i, F90_DPTR_LBOUND_G(ad), 
                              DPTR_UBOUND_G(ad), 1);
    SET_DIM_PTRS(rd, rs, i - 1);
    I8(__fort_set_section)(rc, i, rs, i, F90_DPTR_LBOUND_G(rd), 
                              DPTR_UBOUND_G(rd), 1);
  }

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

  /* move upper section to lower */

  SET_DIM_PTRS(ad, as, dim - 1);
  SET_DIM_PTRS(rd, rs, dim - 1);

  I8(__fort_set_section)(ac, dim, as, dim, F90_DPTR_LBOUND_G(ad) + sabs, 
                          DPTR_UBOUND_G(ad), 1);
  I8(__fort_finish_section)((ac));
  I8(__fort_set_section)(rc, dim, rs, dim, F90_DPTR_LBOUND_G(rd), 
                          DPTR_UBOUND_G(rd) - sabs, 1);
  I8(__fort_finish_section)((rc));

  rp = (char *)rb + DIST_SCOFF_G(rc) * F90_LEN_G(rc);
  ap = (char *)ab + DIST_SCOFF_G(ac) * F90_LEN_G(ac);
  c1 = I8(__fort_copy)(rp, ap, rc, ac, NULL);

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

  /* move lower section to upper */

  I8(__fort_set_section)(ac, dim, as, dim, F90_DPTR_LBOUND_G(ad), 
                          DPTR_UBOUND_G(ad) - (extent - sabs), 1);
  I8(__fort_finish_section)((ac));
  I8(__fort_set_section)(rc, dim, rs, dim, 
                          F90_DPTR_LBOUND_G(rd) + (extent - sabs), 
                          DPTR_UBOUND_G(rd), 1);
  I8(__fort_finish_section)((rc));

  rp = (char *)rb + DIST_SCOFF_G(rc) * F90_LEN_G(rc);
  ap = (char *)ab + DIST_SCOFF_G(ac) * F90_LEN_G(ac);
  c2 = I8(__fort_copy)(rp, ap, rc, ac, NULL);

  c1 = __fort_chain_em_up(c1, c2);
  __fort_doit(c1);
  __fort_frechn(c1);
}

void ENTFTN(CSHIFTSCA, cshiftsca)(DCHAR(rb),    /* result char base */
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
  ENTFTN(CSHIFTS, cshifts)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds);
}

/* 32 bit CLEN version */
void ENTFTN(CSHIFTSC, cshiftsc)(DCHAR(rb),    /* result char base */
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
  ENTFTN(CSHIFTSCA, cshiftsca)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds, 
                             (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab));
}

/* loop over shift array, constructing rank 1 descriptors for each
   corresponding vector section of the array and shifting them the
   selected amount */

void I8(cshift_loop)(void *rb,          /* result base */
                     void *ab,          /* array base */
                     __INT_T *sb,       /* shift base */
                     __INT_T shift_dim, /* dimension to shift */
                     F90_Desc *rs,      /* result descriptor */
                     F90_Desc *as,      /* array descriptor */
                     F90_Desc *ss,      /* shift descriptor */
                     F90_Desc *rc,      /* result subsection descr. */
                     F90_Desc *ac,      /* array subsection descr. */
                     __INT_T soff,      /* shift index */
                     __INT_T loop_dim)  /* current shift dim */
{
  DECL_DIM_PTRS(ad);
  DECL_DIM_PTRS(rd);
  DECL_DIM_PTRS(sd);
  chdr *c1, *c2;
  char *ap, *rp;
  __INT_T aflags, albase, apbase, arepli, ascoff;
  __INT_T rflags, rlbase, rpbase, rrepli, rscoff;
  __INT_T aolb[MAXDIMS], aoub[MAXDIMS];
  __INT_T rolb[MAXDIMS], roub[MAXDIMS];
  __INT_T aflags2, albase2, apbase2, arepli2, ascoff2;
  __INT_T rflags2, rlbase2, rpbase2, rrepli2, rscoff2;
  __INT_T aolb2[MAXDIMS], aoub2[MAXDIMS];
  __INT_T rolb2[MAXDIMS], roub2[MAXDIMS];
  __INT_T array_dim, extent, i, sabs, shift;
  __INT_T ai, al, au, ri, rl, ru, sstr;

  /* shift rank is one dimension less than result/array rank */

  array_dim = loop_dim; /* result/array dim */
  if (loop_dim >= shift_dim)
    ++array_dim;

  SET_DIM_PTRS(rd, rs, array_dim - 1);
  SET_DIM_PTRS(ad, as, array_dim - 1);
  SET_DIM_PTRS(sd, ss, loop_dim - 1);

  ri = F90_DPTR_LBOUND_G(rd);
  ai = F90_DPTR_LBOUND_G(ad);
  sstr = F90_DPTR_SSTRIDE_G(sd) * F90_DPTR_LSTRIDE_G(sd);
  soff += (F90_DPTR_SSTRIDE_G(sd) * F90_DPTR_LBOUND_G(sd) +
           F90_DPTR_SOFFSET_G(sd)) *
          F90_DPTR_LSTRIDE_G(sd);

  /* save descriptor fields affected by set_single */

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

  for (; ri <= DPTR_UBOUND_G(rd); ++ri, ++ai, soff += sstr) {
    I8(__fort_set_single)(rc, rs, array_dim, ri, __SCALAR);
    I8(__fort_set_single)(ac, as, array_dim, ai, __SCALAR);

    if (loop_dim > 1) {
      I8(cshift_loop)(rb, ab, sb, shift_dim,
			    rs, as, ss, rc, ac, soff, loop_dim-1);
    } else {
      shift = sb[soff];

      al = F90_DIM_LBOUND_G(as, shift_dim - 1);
      au = DIM_UBOUND_G(as, shift_dim - 1);
      rl = F90_DIM_LBOUND_G(rs, shift_dim - 1);
      ru = DIM_UBOUND_G(rs, shift_dim - 1);

      /* compute the net positive (left) shift amount */

      extent = au - al + 1;
      if (extent < 0)
        extent = 0;

      sabs = shift % extent;
      if (sabs < 0)
        sabs += extent;

      /* copy directly if net shift amount is zero */

      if (sabs == 0) {
        I8(__fort_set_section)(ac, 1, as, shift_dim, al, au, 1);
        I8(__fort_finish_section)(ac);
        I8(__fort_set_section)(rc, 1, rs, shift_dim, rl, ru, 1);
        I8(__fort_finish_section)(rc);

        rp = (char *)rb + DIST_SCOFF_G(rc) * F90_LEN_G(rs);
        ap = (char *)ab + DIST_SCOFF_G(ac) * F90_LEN_G(as);
        c1 = I8(__fort_copy)(rp, ap, rc, ac, NULL);
      } else {

        /* save descriptor fields affected by set_section */

        aflags2 = F90_FLAGS_G(ac);
        albase2 = F90_LBASE_G(ac);
        apbase2 = DIST_PBASE_G(ac);
        arepli2 = DIST_REPLICATED_G(ac);
        ascoff2 = DIST_SCOFF_G(ac);
        for (i = F90_RANK_G(ac); --i >= 0;) {
          aolb2[i] = DIST_DIM_OLB_G(ac, i);
          aoub2[i] = DIST_DIM_OUB_G(ac, i);
        }
        rflags2 = F90_FLAGS_G(rc);
        rlbase2 = F90_LBASE_G(rc);
        rpbase2 = DIST_PBASE_G(rc);
        rrepli2 = DIST_REPLICATED_G(rc);
        rscoff2 = DIST_SCOFF_G(rc);
        for (i = F90_RANK_G(rc); --i >= 0;) {
          rolb2[i] = DIST_DIM_OLB_G(rc, i);
          roub2[i] = DIST_DIM_OUB_G(rc, i);
        }

        /* move upper section to lower */

        I8(__fort_set_section)(ac, 1, as, shift_dim, al + sabs, au, 1);
        I8(__fort_finish_section)(ac);
        I8(__fort_set_section)(rc, 1, rs, shift_dim, rl, ru - sabs, 1);
        I8(__fort_finish_section)(rc);

        rp = (char *)rb + DIST_SCOFF_G(rc) * F90_LEN_G(rs);
        ap = (char *)ab + DIST_SCOFF_G(ac) * F90_LEN_G(as);
        c1 = I8(__fort_copy)(rp, ap, rc, ac, NULL);

        /* restore descriptor fields */

        F90_FLAGS_P(ac, aflags2);
        F90_LBASE_P(ac, albase2);
        DIST_PBASE_P(ac, apbase2);
        DIST_REPLICATED_P(ac, arepli2);
        DIST_SCOFF_P(ac, ascoff2);
        for (i = F90_RANK_G(ac); --i >= 0;) {
          DIST_DIM_OLB_P(ac, i, aolb2[i]);
          DIST_DIM_OUB_P(ac, i, aoub2[i]);
        }
        DIST_CACHED_P(ac, 0);

        F90_FLAGS_P(rc, rflags2);
        F90_LBASE_P(rc, rlbase2);
        DIST_PBASE_P(rc, rpbase2);
        DIST_REPLICATED_P(rc, rrepli2);
        DIST_SCOFF_P(rc, rscoff2);
        for (i = F90_RANK_G(rc); --i >= 0;) {
          DIST_DIM_OLB_P(rc, i, rolb2[i]);
          DIST_DIM_OUB_P(rc, i, roub2[i]);
        }
        DIST_CACHED_P(rc, 0);

        /* move lower section to upper */

        I8(__fort_set_section)(ac, 1, as, shift_dim, 
                                      al, au - (extent - sabs), 1);
        I8(__fort_finish_section)(ac);
        I8(__fort_set_section)(rc, 1, rs, shift_dim,
				      rl + (extent - sabs), ru, 1);
        I8(__fort_finish_section)(rc);

        rp = (char *)rb + DIST_SCOFF_G(rc) * F90_LEN_G(rs);
        ap = (char *)ab + DIST_SCOFF_G(ac) * F90_LEN_G(as);
        c2 = I8(__fort_copy)(rp, ap, rc, ac, NULL);

        c1 = __fort_chain_em_up(c1, c2);
      }
      __fort_doit(c1);
      __fort_frechn(c1);
    }

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
}

/* result = cshift(array, shift=array, dim) */

void ENTFTN(CSHIFT, cshift)(void *rb,     /* result base */
                            void *ab,     /* array base */
                            __INT_T *sb,  /* shift base */
                            __INT_T *db,  /* dimension to shift */
                            F90_Desc *rs, /* result descriptor */
                            F90_Desc *as, /* array descriptor */
                            F90_Desc *ss, /* shift descriptor */
                            F90_Desc *ds) /* dim descriptor */
{
  DECL_HDR_VARS(ac);
  DECL_HDR_VARS(rc);
  __INT_T dim;

  dim = *db;

#if defined(DEBUG)
  if (__fort_test & DEBUG_CSHF) {
    printf("%d r", GET_DIST_LCPU);
    I8(__fort_show_section)(rs);
    printf("@%x = CSHIFT(a", rb);
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

  I8(cshift_loop)(rb, ab, sb, dim, rs, as, ss, rc, ac,
		    F90_LBASE_G(ss) - 1, F90_RANK_G(ss));
}

void ENTFTN(CSHIFTCA, cshiftca)(DCHAR(rb),    /* result char base */
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
  ENTFTN(CSHIFT, cshift)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds);
}

void ENTFTN(CSHIFTC, cshiftc)(DCHAR(rb),    /* result char base */
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
  ENTFTN(CSHIFTCA, cshiftca)(CADR(rb), CADR(ab), sb, db, rs, as, ss, ds,
                              (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(ab));
}
