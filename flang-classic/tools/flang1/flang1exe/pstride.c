/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Use the pointer analysis Determine when pointers point to aligned
        memory, or contiguous memory, or leftmost-stride-1 memory.  Save
        flow-insensitive pointer target information for use by the back end
 */

#include "gbldefs.h"
#include "global.h"
#include "optimize.h"
#include "symtab.h"
#include "nme.h"
#include "ast.h"
#include "gramtk.h"
#include "pd.h"
#include "extern.h"

static FILE *dfile = NULL;

/*
 * should _findrefs look at pointer A_ID as a pointer dereference?
 */
static int deref = 1;

/*
 * keep a linked list of symbols we've visited,
 * whether we've seen a non-quad-aligned dereference of this symbol,
 * whether we've seen a non-stride-1 dereference of this symbol
 */
typedef struct INFO {
  int next;
  int flags;
  int fistptr;
} INFO;
static INFO *info = NULL;
static int first = 0, count = 0;

#define INFO_QALN 0x1
#define INFO_NON_QALN 0x2
#define INFO_STRIDE1 0x4
#define INFO_NON_STRIDE1 0x8
#define INFO_TARGET 0x10
#define INFO_UNK_TARGET 0x20

#define SEEN_QALN(sptr) (info[sptr].flags & INFO_QALN)
#define SEEN_NON_QALN(sptr) (info[sptr].flags & INFO_NON_QALN)
#define SEEN_STRIDE1(sptr) (info[sptr].flags & INFO_STRIDE1)
#define SEEN_NON_STRIDE1(sptr) (info[sptr].flags & INFO_NON_STRIDE1)
#define SEEN_TARGET(sptr) (info[sptr].flags & INFO_TARGET)
#define SEEN_UNK_TARGET(sptr) (info[sptr].flags & INFO_UNK_TARGET)

#define SET_QALN(sptr) info[sptr].flags |= INFO_QALN
#define SET_NON_QALN(sptr) info[sptr].flags |= INFO_NON_QALN
#define SET_STRIDE1(sptr) info[sptr].flags |= INFO_STRIDE1
#define SET_NON_STRIDE1(sptr) info[sptr].flags |= INFO_NON_STRIDE1
#define SET_TARGET(sptr) info[sptr].flags |= INFO_TARGET
#define SET_UNK_TARGET(sptr) info[sptr].flags |= INFO_UNK_TARGET

/*
 * save Flow-InSensitive pointer-Target information (FIST)
 */
typedef struct FIST {
  int next;
  int ttype, tid; /* target type, target identifier */
} FIST;

static struct {
  FIST *stg_base;
  int stg_avail, stg_size;
} fist = {NULL, 0, 0};

#define FIST_UNK 1
#define FIST_LDYN 2
#define FIST_GDYN 3
#define FIST_NLOC 4
#define FIST_PSYM 5
#define FIST_ISYM 6

#define FIST_TYPE(i) fist.stg_base[i].ttype
#define FIST_ID(i) fist.stg_base[i].tid
#define FIST_NEXT(i) fist.stg_base[i].next

/*
 * save list of FIST for some symbols;
 */
typedef struct FISTLIST {
  int sptr, fistptr;
} FISTLIST;

#define FLIST_SPTR(i) fistlist.stg_base[i].sptr
#define FLIST_PTR(i) fistlist.stg_base[i].fistptr

static struct {
  FISTLIST *stg_base;
  int stg_avail, stg_size;
} fistlist = {NULL, 0, 0};

/*
 * determine whether we have pointer target info for this
 * symbol at this statement; if so, merge that into what we have
 */
static void
do_targets(int stdx, int sptr)
{
  if (!SEEN_UNK_TARGET(sptr)) {
    int c, tag, tid;
    c = 0;
    while (pta_target(stdx, sptr, &tag, &tid)) {
      int t;
      ++c;
      if (tag == FIST_UNK) {
#if DEBUG
        if (DBGBIT(10, 0x200000)) {
          fprintf(gbl.dbgfil, " set UNKTARGET for %d:%s at std %d\n", sptr,
                  SYMNAME(sptr), stdx);
        }
#endif
        SET_UNK_TARGET(sptr);
        return;
      }
      for (t = info[sptr].fistptr; t; t = FIST_NEXT(t)) {
        if (FIST_TYPE(t) == tag && FIST_ID(t) == tid)
          break;
      }
      if (!t) {
        t = fist.stg_avail++;
        OPT_NEED(fist, FIST, 100);
        FIST_TYPE(t) = tag;
        FIST_ID(t) = tid;
        FIST_NEXT(t) = info[sptr].fistptr;
        info[sptr].fistptr = t;
      }
    }
    if (c) {
      SET_TARGET(sptr);
    } else {
      SET_UNK_TARGET(sptr);
#if DEBUG
      if (DBGBIT(10, 0x200000)) {
        fprintf(gbl.dbgfil, " set UNKTARGET (no targets) for %d:%s at std %d\n",
                sptr, SYMNAME(sptr), stdx);
      }
#endif
    }
  }
} /* do_targets */

/*
 * analyze one symbol referenced in this statement
 */
static void
analyze(int sptr, int stdx)
{
  if (sptr && deref) {
    switch (STYPEG(sptr)) {
    case ST_VAR:
    case ST_ARRAY:
      if (ALLOCATTRG(sptr)) {
        /* allocatables are always aligned */
        if (!VISITG(sptr)) {
          info[sptr].flags = 0;
          info[sptr].fistptr = 0;
          info[sptr].next = first;
          first = sptr;
          VISITP(sptr, 1);
        }
        SET_QALN(sptr);
        SET_STRIDE1(sptr);
        if (XBIT(53, 2))
          do_targets(stdx, sptr);
      } else if (deref && POINTERG(sptr) && XBIT(53, 2)) {
#if DEBUG
        if (DBGBIT(10, 0x200000)) {
          fprintf(dfile, "--deref=%d  sptr=%d:%s  std=%d   ", deref, sptr,
                  SYMNAME(sptr), stdx);
          printast(STD_AST(stdx));
          fprintf(dfile, "\n");
          putstdpta(stdx);
        }
#endif
        if (!VISITG(sptr)) {
          info[sptr].flags = 0;
          info[sptr].fistptr = 0;
          info[sptr].next = first;
          first = sptr;
          VISITP(sptr, 1);
        }
        if (!SEEN_NON_QALN(sptr)) {
          if (pta_aligned(stdx, sptr)) {
            SET_QALN(sptr);
          } else {
            SET_NON_QALN(sptr);
          }
        }
        if (!SEEN_NON_STRIDE1(sptr)) {
          if (pta_stride1(stdx, sptr)) {
            SET_STRIDE1(sptr);
          } else {
            SET_NON_STRIDE1(sptr);
          }
        }
        if (XBIT(53, 2))
          do_targets(stdx, sptr);
      }
      break;
    case ST_MEMBER:
      if (ALLOCATTRG(sptr)) {
        /* allocatables are always aligned */
        if (!VISITG(sptr)) {
          info[sptr].flags = 0;
          info[sptr].fistptr = 0;
          info[sptr].next = first;
          first = sptr;
          VISITP(sptr, 1);
        }
        SET_QALN(sptr);
        SET_STRIDE1(sptr);
      }
      else if (deref && POINTERG(sptr) && XBIT(53, 2)) {
        if (!VISITG(sptr)) {
          info[sptr].flags = 0;
          info[sptr].fistptr = 0;
          info[sptr].next = first;
          first = sptr;
          VISITP(sptr, 1);
        }
        if (!SEEN_NON_QALN(sptr)) {
          if (pta_aligned(stdx, sptr)) {
            SET_QALN(sptr);
          } else {
            SET_NON_QALN(sptr);
          }
        }
        if (!SEEN_NON_STRIDE1(sptr)) {
          if (pta_stride1(stdx, sptr)) {
            SET_STRIDE1(sptr);
          } else {
            SET_NON_STRIDE1(sptr);
          }
        }
        if (XBIT(53, 2))
          do_targets(stdx, sptr);
      }
      break;
    default:;
    }
  }
} /* analyze */

/*
 * recursive, depth-first (parent, then all children) traversal of expression
 * tree.
 * special handling for some types of expressions, like procedure calls.
 */
static int
_findrefs(int astx, int *pstdx)
{
  int savederef;
  int asd, ndim, i;
  int args, argcnt, a;

  savederef = deref;

  switch (A_TYPEG(astx)) {
  case A_ID:
    analyze(A_SPTRG(astx), *pstdx);
    break;
  case A_ALLOC:
    /* the object being allocated/deallocated is not interesting */
    deref = 0;
    ast_traverse(A_SRCG(astx), _findrefs, NULL, pstdx);
    /* the status, if given, is interesting */
    deref = 1;
    if (A_LOPG(astx))
      ast_traverse(A_LOPG(astx), _findrefs, NULL, pstdx);
    deref = savederef;
    return 1;
  case A_SUBSCR:
    ast_traverse(A_LOPG(astx), _findrefs, NULL, pstdx);
    /* all subscripts are interesting */
    asd = A_ASDG(astx);
    ndim = ASD_NDIM(asd);
    deref = 1;
    for (i = 0; i < ndim; ++i)
      ast_traverse(ASD_SUBS(asd, i), _findrefs, NULL, pstdx);
    deref = savederef;
    return 1;
  case A_SUBSTR:
    ast_traverse(A_LOPG(astx), _findrefs, NULL, pstdx);
    /* all substring arguments are interesting */
    deref = 1;
    if (A_LEFTG(astx))
      ast_traverse(A_LEFTG(astx), _findrefs, NULL, pstdx);
    if (A_RIGHTG(astx))
      ast_traverse(A_RIGHTG(astx), _findrefs, NULL, pstdx);
    deref = savederef;
    return 1;
  case A_ICALL:
    /* intrinsic call, see if it is ptr assignment */
    if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
      /* pointer assignment, 1st argument is not interesting */
      args = A_ARGSG(astx);
      deref = 0;
      ast_traverse(ARGT_ARG(args, 0), _findrefs, NULL, pstdx);
      deref = 1;
      ast_traverse(ARGT_ARG(args, 1), _findrefs, NULL, pstdx);
      deref = savederef;
      return 1;
    } else if (A_OPTYPEG(astx) == I_NULLIFY) {
      /* pointer nullify */
      args = A_ARGSG(astx);
      deref = 0;
      ast_traverse(ARGT_ARG(args, 0), _findrefs, NULL, pstdx);
      deref = savederef;
      return 1;
    }
    break;
  case A_CALL:
  case A_FUNC:
    /* look at any expression arguments */
    args = A_ARGSG(astx);
    argcnt = A_ARGCNTG(astx);
    for (a = 0; a < argcnt; ++a) {
      deref = 0;
      ast_traverse(ARGT_ARG(args, a), _findrefs, NULL, pstdx);
    }
    deref = savederef;
    return 1;
  }

  deref = 1;
  return 0;
} /* _findrefs */

/*
 * go through all statements, visit all expressions
 * for any expression, find references to any pointers.
 * determine whether any of those references might NOT be to
 * aligned memory, or contiguous memory, or leftmost-stride-1 memory.
 */
void
pstride_analysis(void)
{
  int stdx, astx, sptr;
#if DEBUG
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
#endif
  NEW(info, INFO, stb.stg_avail);
  if (XBIT(53, 2)) {
    points_to(); /* pointsto.c */
    OPT_ALLOC(fist, FIST, 100);
    fist.stg_avail = 1;
  }
  first = 0;
  for (sptr = stb.firstosym; sptr < stb.stg_avail; ++sptr)
    VISITP(sptr, 0);
#if DEBUG
  if (DBGBIT(10, 0x400000)) {
    dstdpa();
  }
#endif
  /* go through all statements, go through all expressions */
  for (stdx = STD_NEXT(0); stdx != 0; stdx = STD_NEXT(stdx)) {
    astx = STD_AST(stdx);
    ast_visit(1, 1);
    ast_traverse(astx, _findrefs, NULL, &stdx);
    ast_unvisit();
  }
  if (XBIT(53, 2)) {
    f90_fini_pointsto(); /* pointsto.c */
  }
  count = 0;
  for (sptr = first; sptr > 0; sptr = info[sptr].next) {
    int ptr, sdsc;
    VISITG(sptr) = 0;
    if (SEEN_QALN(sptr) && !SEEN_NON_QALN(sptr)) {
      ptr = MIDNUMG(sptr);
      if (ptr)
        TQALNP(ptr, 1);
    }
    if (SEEN_STRIDE1(sptr) && !SEEN_NON_STRIDE1(sptr)) {
      SDSCS1P(sptr, 1);
      sdsc = SDSCG(sptr);
      if (sdsc)
        SDSCS1P(sdsc, 1);
    }
    if (SEEN_TARGET(sptr) && !SEEN_UNK_TARGET(sptr)) {
      ++count;
    }
#if DEBUG
    if (DBGBIT(10, 0x200000)) {
      dsym(sptr);
    }
#endif
  }
  if (XBIT(53, 2) && count) {
    OPT_ALLOC(fistlist, FISTLIST, count + 1);
    fistlist.stg_avail = 1;
    for (sptr = first; sptr > 0; sptr = info[sptr].next) {
      if (SEEN_TARGET(sptr) && !SEEN_UNK_TARGET(sptr)) {
        FLIST_SPTR(fistlist.stg_avail) = sptr;
        FLIST_PTR(fistlist.stg_avail) = info[sptr].fistptr;
#if DEBUG
        if (DBGBIT(10, 0x200000)) {
          int t;
          fprintf(gbl.dbgfil, "FIS targets for %s :", SYMNAME(sptr));
          for (t = FLIST_PTR(fistlist.stg_avail); t; t = FIST_NEXT(t)) {
            switch (FIST_TYPE(t)) {
            case FIST_UNK:
              fprintf(gbl.dbgfil, " unk");
              break;
            case FIST_LDYN:
              fprintf(gbl.dbgfil, " ldyn(%d)", FIST_ID(t));
              break;
            case FIST_GDYN:
              fprintf(gbl.dbgfil, " gdyn(%d)", FIST_ID(t));
              break;
            case FIST_NLOC:
              fprintf(gbl.dbgfil, " nloc(%d)", FIST_ID(t));
              break;
            case FIST_PSYM:
              fprintf(gbl.dbgfil, " %d:%s", FIST_ID(t), SYMNAME(FIST_ID(t)));
              break;
            case FIST_ISYM:
              fprintf(gbl.dbgfil, " %d:%s?", FIST_ID(t), SYMNAME(FIST_ID(t)));
              break;
            default:
              fprintf(gbl.dbgfil, " ??%d", FIST_ID(t));
              break;
            }
          }
          fprintf(gbl.dbgfil, "\n");
        }
#endif
        ++fistlist.stg_avail;
      }
    }
  }
  FREE(info);
  first = 0;
  count = 0;
} /* pstride_analysis */

/*
 * put any discovered flow-insensitive pointer target information
 * out to the information file
 */
void
lower_pstride_info(FILE *lowerfile)
{
  int f, t, sptr;
  if (fistlist.stg_base == NULL || fist.stg_base == NULL) {
    return;
  }
  for (f = 1; f < fistlist.stg_avail; ++f) {
    for (t = FLIST_PTR(f); t; t = FIST_NEXT(t)) {
      sptr = FLIST_SPTR(f);
      if (sptr && MIDNUMG(sptr)) {
        switch (FIST_TYPE(t)) {
        case FIST_UNK:
        default:
          fprintf(lowerfile, "info:%d T type:1 id:0\n", MIDNUMG(sptr));
          break;
        case FIST_LDYN:
          fprintf(lowerfile, "info:%d T type:2 id:%d\n", MIDNUMG(sptr),
                  FIST_ID(t));
          break;
        case FIST_GDYN:
          fprintf(lowerfile, "info:%d T type:3 id:%d\n", MIDNUMG(sptr),
                  FIST_ID(t));
          break;
        case FIST_NLOC:
          fprintf(lowerfile, "info:%d T type:4 id:%d\n", MIDNUMG(sptr),
                  FIST_ID(t));
          break;
        case FIST_PSYM:
          fprintf(lowerfile, "info:%d T type:5 id:%d\n", MIDNUMG(sptr),
                  FIST_ID(t));
          break;
        case FIST_ISYM:
          fprintf(lowerfile, "info:%d T type:6 id:%d\n", MIDNUMG(sptr),
                  FIST_ID(t));
          break;
        }
      }
    }
  }
} /* lower_pstride_info */

void
fini_pstride_analysis(void)
{
  if (fist.stg_base) {
    OPT_FREE(fist);
    fist.stg_size = 0;
    fist.stg_avail = 0;
  }
  if (fistlist.stg_base) {
    OPT_FREE(fistlist);
    fistlist.stg_size = 0;
    fistlist.stg_avail = 0;
  }
} /* fini_pstride_analysis */
