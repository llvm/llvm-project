/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/* dbug.c -- runtime internal debug routines */

#include "stdioInterf.h"
#include "fioMacros.h"

#include "type.h"

#include "fort_vars.h"

#ifndef DESC_I8
void ENTFTN(SET_TEST, set_test)(__INT_T *t) { __fort_test = *t; }
#endif

void ENTFTN(ABORTA, aborta)(DCHAR(msg), F90_Desc *msg_s DCLEN64(msg))
{
  char ch;
  ch = CADR(msg)[CLEN(msg)];
  CADR(msg)[CLEN(msg)] = 0;
  __fort_abort(CADR(msg));
  CADR(msg)[CLEN(msg)] = ch;
}

/* 32 bit CLEN version */
void ENTFTN(ABORT, abort)(DCHAR(msg), F90_Desc *msg_s DCLEN(msg))
{
  ENTFTN(ABORTA, aborta)(CADR(msg), msg_s, (__CLEN_T)CLEN(msg));
}

#if !defined(DESC_I8)
void
__fort_print_scalar(void *adr, dtype kind)
{
  if (adr == NULL) {
    fprintf(__io_stderr(), "nil");
    return;
  }
  if (!ISPRESENT(adr) || (char *)adr == ABSENTC) {
    fprintf(__io_stderr(), "absent");
    return;
  }
  switch (kind) {
  case __INT1:
    fprintf(__io_stderr(), "%d", *(__INT1_T *)adr);
    break;
  case __INT2:
    fprintf(__io_stderr(), "%d", *(__INT2_T *)adr);
    break;
  case __INT4:
    fprintf(__io_stderr(), "%d", *(__INT4_T *)adr);
    break;
  case __INT8:
    fprintf(__io_stderr(), "%ld", *(__INT8_T *)adr);
    break;
  case __REAL4:
    fprintf(__io_stderr(), "%g", *(__REAL4_T *)adr);
    break;
  case __REAL8:
    fprintf(__io_stderr(), "%g", *(__REAL8_T *)adr);
    break;
  case __REAL16:
    fprintf(__io_stderr(), "%lg", *(__REAL16_T *)adr);
    break;
  case __CPLX8:
    fprintf(__io_stderr(), "(%g,%g)", ((__CPLX8_T *)adr)->r,
            ((__CPLX8_T *)adr)->i);
    break;
  case __CPLX16:
    fprintf(__io_stderr(), "(%g,%g)", ((__CPLX16_T *)adr)->r,
            ((__CPLX16_T *)adr)->i);
    break;
  case __CPLX32:
    fprintf(__io_stderr(), "(%lg,%lg)", ((__CPLX32_T *)adr)->r,
            ((__CPLX32_T *)adr)->i);
    break;
  case __LOG1:
    fprintf(__io_stderr(),
            *(__LOG1_T *)adr & GET_DIST_MASK_LOG1 ? ".TRUE." : ".FALSE.");
    break;
  case __LOG2:
    fprintf(__io_stderr(),
            *(__LOG2_T *)adr & GET_DIST_MASK_LOG2 ? ".TRUE." : ".FALSE.");
    break;
  case __LOG4:
    fprintf(__io_stderr(),
            *(__LOG4_T *)adr & GET_DIST_MASK_LOG4 ? ".TRUE." : ".FALSE.");
    break;
  case __LOG8:
    fprintf(__io_stderr(),
            *(__LOG8_T *)adr & GET_DIST_MASK_LOG8 ? ".TRUE." : ".FALSE.");
    break;
  case __SHORT:
    fprintf(__io_stderr(), "%d", *(__SHORT_T *)adr);
    break;
  case __CINT:
    fprintf(__io_stderr(), "%d", *(__CINT_T *)adr);
    break;
  case __FLOAT:
    fprintf(__io_stderr(), "%g", *(__FLOAT_T *)adr);
    break;
  case __DOUBLE:
    fprintf(__io_stderr(), "%g", *(__DOUBLE_T *)adr);
    break;
  case __STR:
    fprintf(__io_stderr(), "'%c'", *(char *)adr);
    break;
  default:
    fprintf(__io_stderr(), "%x", *(int *)adr);
  }
}

void
__fort_show_scalar(void *adr, dtype kind)
{
  fprintf(__io_stderr(), "%s=", GET_DIST_TYPENAMES(kind));
  __fort_print_scalar(adr, kind);
}
#endif

void I8(__fort_show_index)(__INT_T rank, __INT_T *index)
{
  int i;

  if (index != NULL) {
    fprintf(__io_stderr(), "(");
    for (i = 0; i < rank; i++) {
      if (i > 0)
        fprintf(__io_stderr(), ",");
      fprintf(__io_stderr(), "%d", index[i]);
    }
    fprintf(__io_stderr(), ")");
  } else
    fprintf(__io_stderr(), "nil");
}

void I8(__fort_show_section)(F90_Desc *d)
{
  DECL_DIM_PTRS(dd);
  __INT_T dx;

  if (ISSEQUENCE(d)) {
    fprintf(__io_stderr(), "SEQUENCE");
    return;
  } else if (ISSCALAR(d)) {
    fprintf(__io_stderr(), "SCALAR");
    return;
  } else if (F90_TAG_G(d) != __DESC) {
    fprintf(__io_stderr(), "not a descriptor\n");
    return;
  }

  fprintf(__io_stderr(), "(");
  for (dx = 0; dx < F90_RANK_G(d); ++dx) {
    if (dx > 0)
      fprintf(__io_stderr(), ",");
    SET_DIM_PTRS(dd, d, dx);
    if (F90_DPTR_LBOUND_G(dd) != 1)
      fprintf(__io_stderr(), "%d:", F90_DPTR_LBOUND_G(dd));
    fprintf(__io_stderr(), "%d", DPTR_UBOUND_G(dd));
  }
  fprintf(__io_stderr(), ")[%d]", F90_GSIZE_G(d));
}

#if !defined(DESC_I8)
static const char *intentnames[4] = {"INOUT", "IN", "OUT", "??"};

static const char *specnames[4] = {"OMITTED", "PRESCRIPTIVE", "DESCRIPTIVE",
                                   "TRANSCRIPTIVE"};
#endif
static const char *dfmtabbrev[] = {"*", "BLK", "BLKK", "CYC", "CYCK", "GENB",
                                   "IND"};

#if !defined(DESC_I8)
void
__fort_show_flags(__INT_T flags)
{
  _io_intent intent;
  _io_spec dist_target_spec;
  _io_spec dist_format_spec;
  _io_spec align_target_spec;

  fprintf(__io_stderr(), "flags=0x%x", flags);
  if (flags & __ASSUMED_SIZE)
    fprintf(__io_stderr(), ", ASSUMED SIZE");
  if (flags & __SEQUENCE)
    fprintf(__io_stderr(), ", SEQUENCE");
  if (flags & __ASSUMED_SHAPE)
    fprintf(__io_stderr(), ", ASSUMED SHAPE");
  if (flags & __SAVE)
    fprintf(__io_stderr(), ", SAVE");
  if (flags & __NO_OVERLAPS)
    fprintf(__io_stderr(), ", NO OVERLAPS");
  intent = (_io_intent)(flags >> __INTENT_SHIFT & __INTENT_MASK);
  if (intent) {
    fprintf(__io_stderr(), ", INTENT(%s)", intentnames[intent]);
  }
  dist_target_spec =
      (_io_spec)(flags >> __DIST_TARGET_SHIFT & __DIST_TARGET_MASK);
  dist_format_spec =
      (_io_spec)(flags >> __DIST_FORMAT_SHIFT & __DIST_FORMAT_MASK);
  align_target_spec =
      (_io_spec)(flags >> __ALIGN_TARGET_SHIFT & __ALIGN_TARGET_MASK);
  if (align_target_spec) {
    fprintf(__io_stderr(), ", %s ALIGN-TARGET", specnames[align_target_spec]);
  }
  if (flags & __IDENTITY_MAP)
    fprintf(__io_stderr(), ", IDENTITY MAP");
  if (flags & __INHERIT)
    fprintf(__io_stderr(), ", INHERIT");
  if (dist_target_spec | dist_format_spec) {
    fprintf(__io_stderr(), ", %s DIST-FORMAT, %s DIST-TARGET",
            specnames[dist_format_spec], specnames[dist_target_spec]);
  }
  if (flags & __DIST_TARGET_AXIS)
    fprintf(__io_stderr(), ", DIST-TARGET AXIS");
  if (flags & __ASSUMED_OVERLAPS)
    fprintf(__io_stderr(), ", ASSUMED_OVERLAPS");
  if (flags & __SECTZBASE)
    fprintf(__io_stderr(), ", SECTZBASE");
  if (flags & __BOGUSBOUNDS)
    fprintf(__io_stderr(), ", BOGUSBOUNDS");
  if (flags & __DYNAMIC)
    fprintf(__io_stderr(), ", DYNAMIC");
  if (flags & __TEMPLATE)
    fprintf(__io_stderr(), ", TEMPLATE");
  if (flags & __LOCAL)
    fprintf(__io_stderr(), ", LOCAL");
  if (flags & __OFF_TEMPLATE)
    fprintf(__io_stderr(), ", OFF TEMPLATE");
  if (flags & __NOT_COPIED)
    fprintf(__io_stderr(), ", NOT COPIED");
  if (flags & __SEQUENTIAL_SECTION)
    fprintf(__io_stderr(), ", SEQUENTIAL_SECTION");
}
#endif

void ENTFTN(SHOW, show)(void *b, F90_Desc *d)
{
  DECL_HDR_PTRS(t);
  DECL_DIM_PTRS(td);
  DECL_DIM_PTRS(dd);
  proc *p;
  procdim *pd;
  __INT_T dx, px, tx;

  I8(__fort_show_section)(d);
  fprintf(__io_stderr(), "@%p F90_Desc@%p rank=%d %s len=%d\n", b, d,
          F90_RANK_G(d), GET_DIST_TYPENAMES(F90_KIND_G(d)), F90_LEN_G(d));

  fprintf(__io_stderr(), "lsize=%d pbase=%d lbase=%d scoff=%d\n",
          F90_LSIZE_G(d), DIST_PBASE_G(d), F90_LBASE_G(d), DIST_SCOFF_G(d));

#if !defined(DESC_I8)
  __fort_show_flags(F90_FLAGS_G(d));
#endif
  fprintf(__io_stderr(), "\n");

  if (F90_RANK_G(d) > 0) {
    fprintf(__io_stderr(), "dim lbnd ubnd  olb  oub   no   po  lab  uab"
                             " lstr loff sstr soff astr aoff\n");
    for (dx = 1; dx <= F90_RANK_G(d); ++dx) {
      SET_DIM_PTRS(dd, d, dx - 1);
      fprintf(__io_stderr(),
              "%3d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d\n", dx,
              F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd), DIST_DPTR_OLB_G(dd),
              DIST_DPTR_OUB_G(dd), DIST_DPTR_NO_G(dd), DIST_DPTR_PO_G(dd),
              DIST_DPTR_LAB_G(dd), DIST_DPTR_UAB_G(dd), F90_DPTR_LSTRIDE_G(dd),
              DIST_DPTR_LOFFSET_G(dd), F90_DPTR_SSTRIDE_G(dd),
              F90_DPTR_SOFFSET_G(dd), DIST_DPTR_ASTRIDE_G(dd),
              DIST_DPTR_AOFFSET_G(dd));
    }

    fprintf(__io_stderr(), "dim   tx tstr toff cost  map olap sect\n");
    for (dx = 1; dx <= F90_RANK_G(d); ++dx) {
      SET_DIM_PTRS(dd, d, dx - 1);
      fprintf(__io_stderr(), "%3d%5d%5d%5d%5d%5d%5d%5d\n", dx,
              DIST_DPTR_TAXIS_G(dd), DIST_DPTR_TSTRIDE_G(dd),
              DIST_DPTR_TOFFSET_G(dd), DIST_DPTR_COFSTR_G(dd),
              (DIST_MAPPED_G(d) >> (dx - 1)) & 1,
              ((DIST_NONSEQUENCE_G(d) >> (dx - 1)) & NONSEQ_OVERLAP) != 0,
              ((DIST_NONSEQUENCE_G(d) >> (dx - 1)) & NONSEQ_SECTION) != 0);
    }

    fprintf(__io_stderr(),
            "dim  tlb  tub dfmt blck cycl  clb  cno   px pcrd pshp pstr\n");
    for (dx = 1; dx <= F90_RANK_G(d); ++dx) {
      SET_DIM_PTRS(dd, d, dx - 1);
      fprintf(__io_stderr(), "%3d%5d%5d%5s%5d%5d%5d%5d%5d%5d%5d%5d\n", dx,
              DIST_DPTR_TLB_G(dd), DIST_DPTR_TUB_G(dd), dfmtabbrev[DFMT(d, dx)],
              DIST_DPTR_BLOCK_G(dd), DIST_DPTR_CYCLE_G(dd), DIST_DPTR_CLB_G(dd),
              DIST_DPTR_CNO_G(dd), DIST_DPTR_PAXIS_G(dd), DIST_DPTR_PCOORD_G(dd),
              DIST_DPTR_PSHAPE_G(dd), DIST_DPTR_PSTRIDE_G(dd));
    }

    if (DIST_CACHED_G(d) != 0) {
      fprintf(__io_stderr(), "dim   cl   cn   cs clof clos\n");
      for (dx = 1; dx <= F90_RANK_G(d); ++dx) {
        if ((DIST_CACHED_G(d) >> (dx - 1)) & 1) {
          SET_DIM_PTRS(dd, d, dx - 1);
          fprintf(__io_stderr(), "%3d%5d%5d%5d%5d%5d\n", dx,
                  DIST_DPTR_CL_G(dd), DIST_DPTR_CN_G(dd), DIST_DPTR_CS_G(dd),
                  DIST_DPTR_CLOF_G(dd), DIST_DPTR_CLOS_G(dd));
        }
      }
    }
  }

  t = DIST_ALIGN_TARGET_G(d);
  if (t != d) {
    fprintf(__io_stderr(), "align-target@%x rank=%d pbase=%d\n", t->tag,
            F90_RANK_G(t), DIST_PBASE_G(t));
#if !defined(DESC_I8)
    __fort_show_flags(F90_FLAGS_G(t));
#endif
    fprintf(__io_stderr(), "\n");

    if (F90_RANK_G(t) > 0) {
      fprintf(__io_stderr(), "dim lbnd ubnd dfmt blck cycl  clb  cno"
                               "   px pcrd pshp pstr sngl info\n");
      for (tx = 1; tx <= F90_RANK_G(t); ++tx) {
        SET_DIM_PTRS(td, t, tx - 1);
        fprintf(__io_stderr(), "%3d%5d%5d%5s%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d\n",
                tx, F90_DPTR_LBOUND_G(td), DPTR_UBOUND_G(td),
                dfmtabbrev[DFMT(t, tx)], DIST_DPTR_BLOCK_G(td),
                DIST_DPTR_CYCLE_G(td), DIST_DPTR_CLB_G(td), DIST_DPTR_CNO_G(td),
                DIST_DPTR_PAXIS_G(td), DIST_DPTR_PCOORD_G(td),
                DIST_DPTR_PSHAPE_G(td), DIST_DPTR_PSTRIDE_G(td),
                (DIST_SINGLE_G(d) >> (tx - 1)) & 1, DIST_INFO_G(d, tx - 1));
      }
    }
  }

  p = DIST_DIST_TARGET_G(d);
  fprintf(__io_stderr(), "dist-target@%x rank=%d size=%d base=%d\n", p->tag,
          p->rank, p->size, p->base);
#if !defined(DESC_I8)
  __fort_show_flags(p->flags);
#endif
  fprintf(__io_stderr(), "\n");

  if (p->rank > 0) {
    fprintf(__io_stderr(), "dim shape stride coord repl\n");
    for (px = 1; px <= p->rank; ++px) {
      pd = &p->dim[px - 1];
      fprintf(__io_stderr(), "%3d%6d%7d%6d%5d\n", px, pd->shape, pd->stride,
              pd->coord, (DIST_REPLICATED_G(d) >> (px - 1)) & 1);
    }
  }
}

#if (defined(DESC_I8) && defined(__PGLLVM__)) || (!defined(DESC_I8) && !defined(__PGLLVM__))
void ENTF90COMN(SHOW_, show_)(void *b, F90_Desc *d)
{
  DECL_DIM_PTRS(dd);
  __INT_T dx;
  OBJECT_DESC *dest = (OBJECT_DESC *)d;
  TYPE_DESC *dest_td;

  if (F90_TAG_G(d) != __POLY && F90_TAG_G(d) != __DESC) {
    dest_td = (dest->type) ? dest->type : (TYPE_DESC *)d;
    d = (F90_Desc *)dest_td;
    fprintf(__io_stderr(), "@%p ", b);
  }

  if (F90_TAG_G(d) == __POLY) {
    fprintf(__io_stderr(), "@%p ", b);
    I8(__fort_dump_type)((TYPE_DESC *)d);
    return;
  }
  I8(__fort_show_section)(d);
  if (F90_TAG_G(d) != __DESC) {
    fprintf(__io_stderr(), "\n");
    return;
  }
  fprintf(__io_stderr(), "@%p F90_Desc@%p rank=%d %s len=%d\n", b, d,
          F90_RANK_G(d), GET_DIST_TYPENAMES(F90_KIND_G(d)), F90_LEN_G(d));

  fprintf(__io_stderr(), "lsize=%d pbase=%d lbase=%d scoff=%d\n",
          F90_LSIZE_G(d), DIST_PBASE_G(d), F90_LBASE_G(d), DIST_SCOFF_G(d));

#if !defined(DESC_I8)
  __fort_show_flags(F90_FLAGS_G(d));
#endif
  fprintf(__io_stderr(), "\n");

  if (F90_RANK_G(d) > 0) {
    fprintf(__io_stderr(), "dim    lbnd    ubnd     ext"
                             "    lstr    sstr    soff\n");
    for (dx = 1; dx <= F90_RANK_G(d); ++dx) {
      SET_DIM_PTRS(dd, d, dx - 1);
      fprintf(__io_stderr(), "%3d %7d %7d %7d %7d %7d %7d\n", dx,
              F90_DPTR_LBOUND_G(dd), DPTR_UBOUND_G(dd), F90_DPTR_EXTENT_G(dd),
              F90_DPTR_LSTRIDE_G(dd), F90_DPTR_SSTRIDE_G(dd),
              F90_DPTR_SOFFSET_G(dd));
    }
  }
}
#endif /* (defined(DESC_I8) && defined(__PGLLVM__)) || (!defined(DESC_I8) && !defined(__PGLLVM__)) */

void I8(__fort_describe)(char *b, F90_Desc *d)
{
  __INT_T dx;

  if (ISSEQUENCE(d)) {
    fprintf(__io_stderr(), "sequence %s at %p = ",
            GET_DIST_TYPENAMES(TYPEKIND(d)), b);
#if !defined(DESC_I8)
    __fort_print_scalar(b, (int)TYPEKIND(d));
#endif
    fprintf(__io_stderr(), "\n");
    return;
  } else if (ISSCALAR(d)) {
    fprintf(__io_stderr(), "scalar %s at %p = ",
            GET_DIST_TYPENAMES(TYPEKIND(d)), b);
#if !defined(DESC_I8)
    __fort_print_scalar(b, (int)TYPEKIND(d));
#endif
    fprintf(__io_stderr(), "\n");
    return;
  } else if (F90_TAG_G(d) != __DESC) {
    fprintf(__io_stderr(), "not a descriptor\n");
    return;
  }

  if (~F90_FLAGS_G(d) & __TEMPLATE) {
    fprintf(__io_stderr(), "%s a_%x(", GET_DIST_TYPENAMES(F90_KIND_G(d)), d->tag);
    for (dx = 0; dx < F90_RANK_G(d); ++dx) {
      if (dx > 0)
        fprintf(__io_stderr(), ",");
      if (F90_DIM_LBOUND_G(d, dx) != 1)
        fprintf(__io_stderr(), "%d:", F90_DIM_LBOUND_G(d, dx));
      fprintf(__io_stderr(), "%d", DIM_UBOUND_G(d, dx));
    }
    fprintf(__io_stderr(), ") at %p\n", b);
    fprintf(__io_stderr(), "!hpf$ shadow a_%x(", d->tag);
    for (dx = 0; dx < F90_RANK_G(d); ++dx) {
      if (dx > 0)
        fprintf(__io_stderr(), ",");
      fprintf(__io_stderr(), "%d:%d", DIST_DIM_NO_G(d, dx),
              DIST_DIM_PO_G(d, dx));
    }
    fprintf(__io_stderr(), ")\n");
    fprintf(__io_stderr(), "local shape (");
    for (dx = 0; dx < F90_RANK_G(d); ++dx) {
      if (dx > 0)
        fprintf(__io_stderr(), ",");
      if (DIST_DIM_LAB_G(d, dx) != 1)
        fprintf(__io_stderr(), "%d:", DIST_DIM_LAB_G(d, dx));
      fprintf(__io_stderr(), "%d", DIST_DIM_UAB_G(d, dx));
    }
    fprintf(__io_stderr(), ")[%d] map (", F90_LSIZE_G(d));
    for (dx = 0; dx < F90_RANK_G(d); ++dx) {
      if (dx > 0)
        fprintf(__io_stderr(), ")+(");
      if (F90_DIM_LSTRIDE_G(d, dx) != 1)
        fprintf(__io_stderr(), "%d*", F90_DIM_LSTRIDE_G(d, dx));
      fprintf(__io_stderr(), "%c", 'i' + dx);
      if (DIST_DIM_LOFFSET_G(d, dx) != 0)
        fprintf(__io_stderr(), "%+d", DIST_DIM_LOFFSET_G(d, dx));
    }
    fprintf(__io_stderr(), ") lbase=%d scoff=%d\n", F90_LBASE_G(d),
            DIST_SCOFF_G(d));
#if !defined(DESC_I8)
    __fort_show_flags(F90_FLAGS_G(d));
#endif
    fprintf(__io_stderr(), "\n");
  }
}

void ENTFTN(DESCRIBE, describe)(void *b, F90_Desc *d)
{
  I8(__fort_describe)(b, d);
}

/* general array section print routine */

static void I8(print_row)(void *ab, __INT_T str, __INT_T cnt, dtype kind)
{
  __INT_T i, *ci;
  __INT1_T *i1;
  __INT2_T *i2;
  __INT4_T *i4;
  __INT8_T *i8;
  __LOG1_T *l1;
  __LOG2_T *l2;
  __LOG4_T *l4;
  __LOG8_T *l8;
  __REAL4_T *r4;
  __REAL8_T *r8;
  __REAL16_T *r16;
  __CPLX8_T *c8;
  __CPLX16_T *c16;
  __CPLX32_T *c32;

  switch (kind) {
  case __CINT:
    ci = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 15) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %4d", ci[i * str]);
    }
    break;
  case __INT1:
    i1 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 15) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %4d", i1[i * str]);
    }
    break;
  case __INT2:
    i2 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 15) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %4d", i2[i * str]);
    }
    break;
  case __INT4:
    i4 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 15) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %4d", i4[i * str]);
    }
    break;
  case __INT8:
    i8 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 15) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %ld", i8[i * str]);
    }
    break;
  case __LOG1:
    l1 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 31) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), l1[i * str] & GET_DIST_MASK_LOG1 ? " T" : " F");
    }
    break;
  case __LOG2:
    l2 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 31) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), l2[i * str] & GET_DIST_MASK_LOG2 ? " T" : " F");
    }
    break;
  case __LOG4:
    l4 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 31) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), l4[i * str] & GET_DIST_MASK_LOG4 ? " T" : " F");
    }
    break;
  case __LOG8:
    l8 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 31) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), l8[i * str] & GET_DIST_MASK_LOG8 ? " T" : " F");
    }
    break;
  case __REAL4:
    r4 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 7) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %g", r4[i * str]);
    }
    break;
  case __REAL8:
    r8 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 7) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %g", r8[i * str]);
    }
    break;
  case __REAL16:
    r16 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 7) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " %lg", r16[i * str]);
    }
    break;
  case __CPLX8:
    c8 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 3) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " (%g,%g)", c8[i * str].r, c8[i * str].i);
    }
    break;
  case __CPLX16:
    c16 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 3) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " (%g,%g)", c16[i * str].r, c16[i * str].i);
    }
    break;
  case __CPLX32:
    c32 = ab;
    for (i = 0; i < cnt; ++i) {
      if (i > 0 && (i & 3) == 0)
        fprintf(__io_stderr(), "\n");
      fprintf(__io_stderr(), " (%lg,%lg)", c32[i * str].r, c32[i * str].i);
    }
    break;
  default:
    __fort_abort("print_local: unsupported type");
  }
}

static void I8(print_loop)(char *b, F90_Desc *d, __INT_T rowdim, __INT_T dim,
                           __INT_T off)
{
  DECL_DIM_PTRS(dd);
  __INT_T cl, clof, cn, k, l, n, u;

  if (dim == rowdim)
    --dim;
  if (dim < 1)
    dim = rowdim;

  SET_DIM_PTRS(dd, d, dim - 1);

  cl = DIST_DPTR_CL_G(dd);
  cn = DIST_DPTR_CN_G(dd);
  clof = DIST_DPTR_CLOF_G(dd);

  for (; cn > 0; --cn, cl += DIST_DPTR_CS_G(dd), clof += DIST_DPTR_CLOS_G(dd)) {

    n = I8(__fort_block_bounds)(d, dim, cl, &l, &u);

    k = off +
        (l * F90_DPTR_SSTRIDE_G(dd) + F90_DPTR_SOFFSET_G(dd) - clof) *
            F90_DPTR_LSTRIDE_G(dd);

    if (dim != rowdim) {
      for (; --n >= 0; k += F90_DPTR_SSTRIDE_G(dd) * F90_DPTR_LSTRIDE_G(dd))
        I8(print_loop)(b, d, rowdim, dim - 1, k);
    } else
      I8(print_row)(b + k*F90_LEN_G(d), 
                          F90_DPTR_SSTRIDE_G(dd)* F90_DPTR_LSTRIDE_G(dd),
                          n, F90_KIND_G(d));
  }
  if (dim == rowdim)
    fprintf(__io_stderr(), "\n");
}

void I8(__fort_print_local)(void *b, F90_Desc *d)
{
  int save_test;

  save_test = __fort_test;
  __fort_test = 0;
  I8(__fort_cycle_bounds)(d);
  if (F90_FLAGS_G(d) & __OFF_TEMPLATE)
    fprintf(__io_stderr(), " -- no local data --\n");
  else
    I8(print_loop)(b, d, Min(2,F90_RANK_G(d)), F90_RANK_G(d), 
                       F90_LBASE_G(d) - 1);
  __fort_test = save_test;
}

/* print a strided vector */

void I8(__fort_print_vector)(char *msg, void *adr, __INT_T str, __INT_T cnt,
                            dtype kind)
{
  fprintf(__io_stderr(), "%d %s\n", GET_DIST_LCPU, msg);
  I8(print_row)(adr, str, cnt, kind);
  fprintf(__io_stderr(), "\n");
}
