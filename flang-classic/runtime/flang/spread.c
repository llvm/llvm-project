/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"

/* spread intrinsic -- copy sections for ncopies into appropriate
   dimensions */

void ENTFTN(SPREAD, spread)(void *rb,           /* result base */
                            void *sb,           /* source base */
                            void *dimb,         /* dimension base */
                            void *ncopiesb,     /* ncopies base */
                            F90_Desc *rd,       /* result descriptor */
                            F90_Desc *sd,       /* source descriptor */
                            F90_Desc *dimd,     /* dimension descriptor */
                            F90_Desc *ncopiesd) /* ncopies descriptor */
{
  char *rp, *sp;
  chdr *c;
  DECL_DIM_PTRS(rdd);
  DECL_HDR_VARS(td);
  int k, dim, ncopies;
  __INT_T i, rank, rx, tx;
  __INT_T olb[MAXDIMS], oub[MAXDIMS];
  __INT_T flags, lbase, pbase, repli, scoff;

  dim = I8(__fort_fetch_int)(dimb, dimd);
  ncopies = I8(__fort_fetch_int)(ncopiesb, ncopiesd);

  /* form temporary descriptor with a scalar subscript in the spread
     dimension */

  rank = F90_RANK_G(rd) - 1;
  __DIST_INIT_SECTION(td, rank, rd);

  /* set up non-spread dimensions */

  for (tx = 0, rx = 1; rx <= F90_RANK_G(rd); ++rx) {
    if (rx != dim) {
      SET_DIM_PTRS(rdd, rd, rx - 1);
      I8(__fort_set_section)(td, ++tx, rd, rx, F90_DPTR_LBOUND_G(rdd), 
                                  DPTR_UBOUND_G(rdd), 1);
    }
  }

  flags = F90_FLAGS_G(td); /* save descriptor fields */
  lbase = F90_LBASE_G(td);
  pbase = DIST_PBASE_G(td);
  repli = DIST_REPLICATED_G(td);
  scoff = DIST_SCOFF_G(td);
  for (i = rank; --i >= 0;) {
    olb[i] = DIST_DIM_OLB_G(td, i);
    oub[i] = DIST_DIM_OUB_G(td, i);
  }

  sp = (char *)sb + DIST_SCOFF_G(sd) * F90_LEN_G(sd);
  SET_DIM_PTRS(rdd, rd, dim - 1);
  for (k = 0; k < ncopies; ++k) {

    /* set scalar subscript in spread dimension */

    I8(__fort_set_single)((td), rd, dim, F90_DPTR_LBOUND_G(rdd) + k, __SCALAR);
    I8(__fort_finish_section)((td));

    rp = (char *)rb + DIST_SCOFF_G(td) * F90_LEN_G(td);
    c = I8(__fort_copy)(rp, sp, td, sd, NULL);
    __fort_doit(c);
    __fort_frechn(c);

    F90_FLAGS_P(td, flags); /* restore descriptor fields */
    F90_LBASE_P(td, lbase);
    DIST_PBASE_P(td, pbase);
    DIST_REPLICATED_P(td, repli);
    DIST_SCOFF_P(td, scoff);
    for (i = rank; --i >= 0;) {
      DIST_DIM_OLB_P(td, i, olb[i]);
      DIST_DIM_OUB_P(td, i, oub[i]);
    }
    DIST_CACHED_P(td, 0);
  }
}

/* spread of a scalar - copy the scalar to a rank 1 array, ignore 'dim' */
void ENTFTN(SPREADSA, spreadsa)(void *rb,           /* result base */
                              void *sb,           /* source base */
                              void *dimb,         /* dimension base */
                              void *ncopiesb,     /* ncopies base */
                              __CLEN_T *szb,       /* sizeof source base */
                              F90_Desc *rd,       /* result descriptor */
                              F90_Desc *sd,       /* source descriptor */
                              F90_Desc *dimd,     /* dimension descriptor */
                              F90_Desc *ncopiesd, /* ncopies descriptor */
                              F90_Desc *szd)      /* sizeof source descriptor */
{
  char *rp;
  int ncopies;
  __CLEN_T size;

  /* we assume that result is replicated and contiguous */

  ncopies = I8(__fort_fetch_int)(ncopiesb, ncopiesd);
  size = *szb;
  rp = (char *)rb;
  while (ncopies-- > 0) {
    __fort_bcopy(rp, sb, size);
    rp = rp + size;
  }
}
/* 32 bit CLEN version */
void ENTFTN(SPREADS, spreads)(void *rb,           /* result base */
                              void *sb,           /* source base */
                              void *dimb,         /* dimension base */
                              void *ncopiesb,     /* ncopies base */
                              __INT_T *szb,       /* sizeof source base */
                              F90_Desc *rd,       /* result descriptor */
                              F90_Desc *sd,       /* source descriptor */
                              F90_Desc *dimd,     /* dimension descriptor */
                              F90_Desc *ncopiesd, /* ncopies descriptor */
                              F90_Desc *szd)      /* sizeof source descriptor */
{
  ENTFTN(SPREADSA, spreadsa)(rb, sb, dimb, ncopiesb, (__CLEN_T *)szb, rd, sd,
         dimd, ncopiesd, szd);
}

void ENTFTN(SPREADCA, spreadca)(DCHAR(rb),         /* result char base */
                              DCHAR(sb),         /* source char base */
                              void *dimb,        /* dimension base */
                              void *ncopiesb,    /* ncopies base */
                              F90_Desc *rd,      /* result descriptor */
                              F90_Desc *sd,      /* source descriptor */
                              F90_Desc *dimd,    /* ncopies descriptor */
                              F90_Desc *ncopiesd /* dimension descriptor */
                              DCLEN64(rb)          /* result char len */
                              DCLEN64(sb))         /* source char len */
{
  ENTFTN(SPREAD,spread)(CADR(rb), CADR(sb), dimb, ncopiesb,
			  rd, sd, dimd, ncopiesd);
}
/* 32 bit CLEN version */
void ENTFTN(SPREADC, spreadc)(DCHAR(rb),         /* result char base */
                              DCHAR(sb),         /* source char base */
                              void *dimb,        /* dimension base */
                              void *ncopiesb,    /* ncopies base */
                              F90_Desc *rd,      /* result descriptor */
                              F90_Desc *sd,      /* source descriptor */
                              F90_Desc *dimd,    /* ncopies descriptor */
                              F90_Desc *ncopiesd /* dimension descriptor */
                              DCLEN(rb)          /* result char len */
                              DCLEN(sb))         /* source char len */
{
  ENTFTN(SPREADCA, spreadca)(CADR(rb), CADR(sb), dimb, ncopiesb, rd, sd, dimd,
         ncopiesd, (__CLEN_T)CLEN(rb), (__CLEN_T)CLEN(sb));
}

/* spread of a character scalar - copy the scalar to a rank 1 array, ignore
 * 'dim' */
void ENTFTN(SPREADCSA,
            spreadcsa)(DCHAR(rb),      /* result char base */
                      DCHAR(sb),      /* source char base */
                      void *dimb,     /* dimension base */
                      void *ncopiesb, /* ncopies base */
                      __CLEN_T *szb,   /* sizeof source base - 0 for spreadcs */
                      F90_Desc *rd,   /* result descriptor */
                      F90_Desc *sd,   /* source descriptor */
                      F90_Desc *dimd, /* ncopies descriptor */
                      F90_Desc *ncopiesd, /* dimension descriptor */
                      F90_Desc *szd       /* sizeof source descriptor */
                      DCLEN64(rb)           /* result char len */
                      DCLEN64(sb))          /* source char len */
{
  __CLEN_T size;
  size = CLEN(sb);
  ENTFTN(SPREADS,spreads)(CADR(rb), CADR(sb), dimb, ncopiesb, &size,
			    rd, sd, dimd, ncopiesd, szd);
}
/* 32 bit CLEN version */
void ENTFTN(SPREADCS,
            spreadcs)(DCHAR(rb),      /* result char base */
                      DCHAR(sb),      /* source char base */
                      void *dimb,     /* dimension base */
                      void *ncopiesb, /* ncopies base */
                      __INT_T *szb,   /* sizeof source base - 0 for spreadcs */
                      F90_Desc *rd,   /* result descriptor */
                      F90_Desc *sd,   /* source descriptor */
                      F90_Desc *dimd, /* ncopies descriptor */
                      F90_Desc *ncopiesd, /* dimension descriptor */
                      F90_Desc *szd       /* sizeof source descriptor */
                      DCLEN(rb)           /* result char len */
                      DCLEN(sb))          /* source char len */
{
  ENTFTN(SPREADCSA, spreadcsa)(CADR(rb), CADR(sb), dimb, ncopiesb,
         (__CLEN_T *)szb, rd, sd, dimd, ncopiesd, szd, (__CLEN_T)CLEN(rb),
         (__CLEN_T)CLEN(sb));
}

/* set up result descriptor for spread intrinsic -- used when the dim
   arg is variable.  the added spread dimension is given a collapsed
   distribution and the remaining dimensions are aligned with the
   corresponding source dimensions.  lbounds are set to 1 and overlap
   allowances are set to 0. */

void ENTFTN(SPREAD_DESCRIPTOR,
            spread_descriptor)(F90_Desc *rd,      /* result descriptor */
                               F90_Desc *sd,      /* source descriptor */
                               __INT_T *dimb,     /* dimension */
                               __INT_T *ncopiesb) /* ncopies base */
{
  DECL_DIM_PTRS(sdd);
  DECL_HDR_PTRS(td);
  __INT_T dim, extent, m, ncopies, offset, rx, sx, tx;

#if defined(DEBUG)
  if (F90_TAG_G(sd) != __DESC)
    __fort_abort("SPREAD: invalid source arg");
#endif

  dim = *dimb;
  if (dim < 1 || dim > F90_RANK_G(sd) + 1)
    __fort_abort("SPREAD: invalid dim");

  ncopies = *ncopiesb;
  if (ncopies < 0)
    ncopies = 0;

  td = DIST_ALIGN_TARGET_G(sd);
  __DIST_INIT_DESCRIPTOR(rd, F90_RANK_G(sd) + 1, F90_KIND_G(sd), F90_LEN_G(sd),
                        F90_FLAGS_G(sd), td);
  for (rx = sx = 1; sx <= F90_RANK_G(sd); ++rx, ++sx) {
    if (sx == dim)
      ++rx;
    SET_DIM_PTRS(sdd, sd, sx - 1);
    extent = F90_DPTR_EXTENT_G(sdd);
    offset = DIST_DPTR_TSTRIDE_G(sdd) * (F90_DPTR_LBOUND_G(sdd) - 1) +
             DIST_DPTR_TOFFSET_G(sdd);

    /* 
     * added &DIST_DIM_GEN_BLOCK_G(td,(DIST_DPTR_TAXIS_G(sdd))-1) arg
     */

    I8(__fort_set_alignment)(rd, rx, 1, extent, DIST_DPTR_TAXIS_G(sdd), 
                                DIST_DPTR_TSTRIDE_G(sdd), offset,
                                &DIST_DIM_GEN_BLOCK_G(td,(DIST_DPTR_TAXIS_G(sdd))-1));
    __DIST_SET_ALLOCATION(rd, rx, 0, 0);
  }
  I8(__fort_set_alignment)(rd, dim, 1, ncopies, 0, 1, 0);
  __DIST_SET_ALLOCATION(rd, dim, 0, 0);
  m = DIST_SINGLE_G(sd);
  for (tx = 1; m > 0; ++tx, m >>= 1) {
    if (m & 1)
      I8(__fort_set_single)(rd, td, tx, DIST_INFO_G(sd, tx - 1), __SINGLE);
  }
  I8(__fort_finish_descriptor)(rd);
}
