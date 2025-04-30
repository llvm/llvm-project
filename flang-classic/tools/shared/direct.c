/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief Directive/pragma support modules
 */

#include "fdirect.h"
#include "pragma.h"
#include "ilidir.h"
#include "miscutil.h"

#if DEBUG
static void dmp_dirset(DIRSET *);
static void dmp_lpprg(int);
#define TR0(s)         \
  if (DBGBIT(1, 1024)) \
    fprintf(gbl.dbgfil, s);
#define TR1(s, a)      \
  if (DBGBIT(1, 1024)) \
    fprintf(gbl.dbgfil, s, a);
#define TR2(s, a, b)   \
  if (DBGBIT(1, 1024)) \
    fprintf(gbl.dbgfil, s, a, b);
#define TR3(s, a, b, c) \
  if (DBGBIT(1, 1024))  \
    fprintf(gbl.dbgfil, s, a, b, c);

#else
#define TR0(s)
#define TR1(s, a)
#define TR2(s, a, b)
#define TR3(s, a, b, c)

#endif
#include "mach.h"

DIRECT direct;

#ifdef FE90

/* For saving the directives for the backends of the native compilers */

typedef struct {
  int beg_line;  /* beginning line # of loop */
  int end_line;  /* ending line # of loop */
  DIRSET change; /* what's changed */
  DIRSET newset; /* dirset state for the routine & loops */
                 /* How an entry in newset is propagated depends on the
                  * type (value vs bit vector) of the entry:
                  * - if value:
                  *      change == nonzero => propagate the value
                  * - if a bit vector:
                  *      change == nonzero => for each bit which is
                  *      set, the new value of the corresponding bit
                  *      position is propagated.  All other bits are
                  *      left untouched.
                  *      The change value is computed as
                  *         change = new ^ old
                  *      The new value of the bit vector is
                  *      computed as
                  *         val = (new & change) | (current & !change)
                  */
} SVDIR;

static struct {
  SVDIR *stgb; /* [0]   is reserved for the routine's directives.
                * [>=1] used for the loops' directives.
                */
  int size;
  int avail; /* avail is actually tracked as direct.lpg.avail;
              * svdir.avail is needed for direct_export()
              */
} svdir = {NULL, 0, 0};

static DIRSET inigbl;

static void update_rou_begin(void);
static void diff_dir(SVDIR *, DIRSET *, DIRSET *);
static void wr_dir(FILE *, SVDIR *);
#endif

/** \brief Initialize directive structure
 *
 * Initialize directive structure which is global for the
 * source file and the structure which is used per routine.
 * must be called just once for the compilation and after all of the
 * command line options have been processed.
 *
 * The initial values of the global structure are extracted from
 * the command line options.  The routine structure is initialized
 * with values from the global structure.
 */
void
direct_init(void)
{

/* Set/clear any xbits for which the command line processing has no
 * effect
 */
  flg.x[8] |= 0x8; /* disable global reg assignment */

  store_dirset(&direct.gbl);

#ifdef FE90
  inigbl = direct.gbl;
#endif
  direct.rou = direct.gbl;
  direct.loop = direct.gbl;
  direct.rou_begin = direct.gbl;

  direct.loop_flag = false; /* seen pragma with loop scope */
  direct.in_loop = false;   /* in loop with pragmas */
  direct.carry_fwd = false;

  direct.avail = 0;
  NEW(direct.stgb, DIRSET, (direct.size = 16));

  direct.lpg.avail = 1;

  NEW(direct.lpg.stgb, LPPRG, (direct.lpg.size = 16));
  BZERO(direct.lpg.stgb, LPPRG, 1);
  direct.lpg_stk.top = 0;
  NEW(direct.lpg_stk.stgb, LPG_STK, (direct.lpg_stk.size = 8));

#ifdef FE90
  direct.dynlpg.avail = 1;
  NEW(direct.dynlpg.stgb, LPPRG, (direct.dynlpg.size = 16));
  if (flg.genilm) {
    /* [0] is reserved for the 'routine' directives */
    NEW(svdir.stgb, SVDIR, (svdir.size = 16));
    BZERO(svdir.stgb, SVDIR, 1); /* init [0] to zero */
  }
  direct.indep = NULL;
  direct.index_reuse_list = NULL;
#endif

}

void
direct_fini()
{
  if (direct.stgb) {
    FREE(direct.stgb);
    direct.avail = direct.size = 0;
  }
  if (direct.lpg.stgb) {
    FREE(direct.lpg.stgb);
    direct.lpg.avail = direct.lpg.size = 0;
  }
  if (direct.lpg_stk.stgb) {
    FREE(direct.lpg_stk.stgb);
    direct.lpg_stk.top = direct.lpg_stk.size = 0;
  }
#ifdef FE90
  if (direct.dynlpg.stgb) {
    FREE(direct.dynlpg.stgb);
    direct.dynlpg.avail = direct.dynlpg.size = 0;
  }
  if (svdir.stgb) {
    FREE(svdir.stgb);
    svdir.avail = svdir.size = 0;
  }
#endif
} /* direct_fini */

/** \brief Re-initialize the routine structure
 *
 * Must be called after the end of a function is processed by semant and
 * before the next function is parsed. For C, this is when the END ilm is
 * written for a function.  For Fortran, this is after a subprogram has been
 * processed by all phases of the compiler.
 *
 * For C, process any symbol-/variable- related pragmas which may have
 * occurred.  Also, save the index into the lpprg table which is the beginning
 * of the function's lpprg segment; mark the end of the segment with an entry
 * whose beg_line is -1.
 */
void
direct_rou_end(void)
{
/* CPLUS also needs to save routine's structure: */
#ifdef FE90
  if (flg.genilm) {
    update_rou_begin();
  }
#endif
  direct.lpg.avail = 1;

  direct.rou = direct.gbl;
  direct.loop = direct.gbl;
  direct.rou_begin = direct.gbl;
  direct.carry_fwd = false;
#ifdef FE90
  direct.dynlpg.avail = 1;
#endif

}

void
direct_loop_enter(void)
{
  if (direct.loop_flag || (direct.carry_fwd && !direct.in_loop)) {
    push_lpprg(gbl.lineno);
#ifdef FE90
    if (flg.genilm) {
      NEEDB(direct.lpg.avail, svdir.stgb, SVDIR, svdir.size,
            direct.lpg.avail + 8);
      diff_dir(&svdir.stgb[direct.lpg.avail - 1],
               &direct.lpg.stgb[direct.lpg.avail - 1].dirset, &direct.loop);
    }
#endif
  }

}

/** \brief Re-initialize the loop structure
 *
 * Must be called after the end of a loop is processed by semant for which
 * loop-scoped pragmas/directives apply.
 */
void
direct_loop_end(int beg_line, int end_line)
{
  int i;
  LPPRG *lpprg;

  if (!direct.in_loop)
    return;
  i = direct.lpg_stk.stgb[direct.lpg_stk.top].dirx;
  lpprg = direct.lpg.stgb + i;
  if (lpprg->beg_line != beg_line)
    return;

/***** pop_lpprg *****/

  TR1("---pop_lpprg: top %d,", direct.lpg_stk.top);
  direct.lpg_stk.top--;
  TR1(" lpprg %d,", i);
  lpprg = direct.lpg.stgb + i;

  lpprg->end_line = end_line;
  TR2(" beg %d, end %d\n", lpprg->beg_line, lpprg->end_line);

  direct.loop = direct.rou;

#ifdef FE90
  if (flg.genilm) {
    svdir.stgb[i].beg_line = lpprg->beg_line;
    svdir.stgb[i].end_line = lpprg->end_line;
  }
#endif

  if (direct.lpg_stk.top == 0) {
    direct.loop_flag = false;
    direct.in_loop = false;
  } else if (XBIT(59, 1)) {
    direct.loop =
        direct.lpg.stgb[direct.lpg_stk.stgb[direct.lpg_stk.top].dirx].dirset;
  } else {
    /*
     * propagate selected directives/pragmas to all nested
     */
    i = direct.lpg_stk.stgb[direct.lpg_stk.top].dirx;
    direct.loop.depchk = direct.lpg.stgb[i].dirset.depchk;
  }
#if DEBUG
  if (DBGBIT(1, 512))
    dmp_lpprg(direct.lpg_stk.stgb[direct.lpg_stk.top + 1].dirx);
#endif

}

#ifdef FE90
/*
 * for the IPA recompile, save the loop pragma directly
 */
void
direct_loop_save()
{
  int i;
  for (i = 1; i < direct.lpg.avail; ++i) {
    NEED(i + 1, svdir.stgb, SVDIR, svdir.size, i + 8);
    diff_dir(&svdir.stgb[i], &direct.lpg.stgb[i].dirset, &direct.loop);
    svdir.stgb[i].beg_line = direct.lpg.stgb[i].beg_line;
    svdir.stgb[i].end_line = direct.lpg.stgb[i].end_line;
  }
} /* direct_loop_save */
#endif

typedef struct xf_tag {
  char *fn; /* name of function */
  int x;    /* which xflag */
  int v;    /* value of xflag */
  struct xf_tag *next;
} XF;

static XF *xf_p = NULL; /* list of function -x ... */
static XF *yf_p = NULL; /* list of function -y ... */

void
direct_xf(char *fn, int x, int v)
{
  XF *xfp;
  /*printf("-xf %s %d 0x%x\n", fn, x, v);*/
  xfp = (XF *)getitem(8, sizeof(XF));
  xfp->next = xf_p;
  xf_p = xfp;
  xfp->fn = fn;
  xfp->x = x;
  xfp->v = v;
}

void
direct_yf(char *fn, int x, int v)
{
  XF *xfp;
  /*printf("-yf %s %d 0x%x\n", fn, x, v);*/
  xfp = (XF *)getitem(8, sizeof(XF));
  xfp->next = yf_p;
  yf_p = xfp;
  xfp->fn = fn;
  xfp->x = x;
  xfp->v = v;
}

/** \brief Load direct.rou for the current function
 *
 * Called after the parse phase is complete; called once per function.  For C
 * this means the call occurs during expand when it sees the ENTRY ilm; for
 * Fortran, this is at the beginning of expand (in main).
 *
 * DIRSET direct.rou_begin represents the state of the directives/pragmas at
 * the beginning of the function.
 *
 * \param func - symbol of current function
 */
void
direct_rou_load(int func)
{
  DIRSET *currdir;
  XF *xfp;
  char *fnp;

  currdir = &direct.rou_begin;

  load_dirset(currdir);

  fnp = SYMNAME(gbl.currsub);
  for (xfp = xf_p; xfp != NULL; xfp = xfp->next) {
    if (strcmp(xfp->fn, fnp) == 0) {
      /*printf("-xf %s %d 0x%x\n", xfp->fn, xfp->x, xfp->v);*/
      set_xflag(xfp->x, xfp->v);
      currdir->x[xfp->x] = flg.x[xfp->x];
    }
  }
  for (xfp = yf_p; xfp != NULL; xfp = xfp->next) {
    if (strcmp(xfp->fn, fnp) == 0) {
      /*printf("-yf %s %d 0x%x\n", xfp->fn, xfp->x, xfp->v);*/
      set_yflag(xfp->x, xfp->v);
      currdir->x[xfp->x] = flg.x[xfp->x];
    }
  }

#ifndef FE90
  /*
   * the optimizer doesn't handle assigned goto's correctly.
   * (Doesn't know where to put loop exit code if you assign
   * goto out of loop)
   */
  if ((gbl.asgnlbls == -1) && (flg.opt >= 2)) {
    error(I_0127_Optimization_level_for_OP1_changed_to_opt_1_OP2, ERR_Informational, 0, SYMNAME(gbl.currsub), "due to assigned goto");
    currdir->opt = flg.opt = 1;
    currdir->vect = flg.vect = 0;
  }
  if (gbl.vfrets) {
    /*
     * temporarily disable optimizations not correctly
     * handle if variable functions occur.
     */
    if (flg.opt >= 2) {
      error(I_0127_Optimization_level_for_OP1_changed_to_opt_1_OP2, ERR_Informational, 0, SYMNAME(gbl.currsub), "due to < > in FORMAT");
      currdir->opt = flg.opt = 1;
      currdir->vect = flg.vect = 0;
    }
    flg.x[8] |= 0x8; /* no globalregs at opt 1 */
  }
#endif

#if DEBUG
  if (DBGBIT(1, 256)) {
    fprintf(gbl.dbgfil, "---dirset for func ");
    fprintf(gbl.dbgfil, "%s\n", SYMNAME(func));
    dmp_dirset(currdir);
  }
#endif

#if (defined(TARGET_X86) || defined(TARGET_LLVM)) && !defined(FE90)
    set_mach(&mach, direct.rou_begin.tpvalue[0]);
#endif

}

void
direct_rou_setopt(int func, int opt)
{
  DIRSET *currdir;
  currdir = &direct.rou_begin;
  flg.opt = opt;
  currdir->opt = opt;
}

void
load_dirset(DIRSET *currdir)
{
  flg.depchk = currdir->depchk;
  flg.opt = currdir->opt;
  flg.vect = currdir->vect;
  BCOPY(flg.tpvalue, currdir->tpvalue, int, TPNVERSION);
  BCOPY(flg.x, currdir->x, int, (INT)sizeof(flg.x) / sizeof(int));
#if DEBUG
  if (DBGBIT(1, 2048))
    dmp_dirset(currdir);
#endif

}

void
store_dirset(DIRSET *currdir)
{
  currdir->depchk = flg.depchk;
  currdir->opt = flg.opt;
  currdir->vect = flg.vect;
  BCOPY(currdir->tpvalue, flg.tpvalue, int, TPNVERSION);
  BCOPY(currdir->x, flg.x, int, (INT)sizeof(flg.x) / sizeof(int));

}

/** \brief OPTIONS statement processed (by scan via semant)
 *
 * These only affect
 * what happens in semant for the 'next' routine.  alter any dirset
 * values which can be altered by OPTIONS.
 *
 * \param restore true if called when restoring effects of OPTIONS
 */
void
dirset_options(bool restore)
{
  if (restore)
    direct.rou_begin.x[70] = direct.gbl.x[70];
  else
    direct.rou_begin.x[70] = flg.x[70];

}

#if DEBUG
static void
dmp_dirset(DIRSET *currdir)
{
#define _FNO(s) ((s) ? "" : "no")
#define _TNO(s) ((s) ? "no" : "")
  fprintf(gbl.dbgfil,
          "   opt=%d,%sdepchk,%sassoc,%stransform,%srecog,%sswpipe,%sstream\n",
          currdir->opt, _FNO(currdir->depchk), _TNO(currdir->vect & 0x4),
          _TNO(currdir->x[19] & 0x8), _TNO(currdir->x[19] & 0x10),
          _TNO(currdir->x[19] & 0x20), _TNO(currdir->x[19] & 0x40));
  fprintf(gbl.dbgfil, "   shortloop:%d", currdir->x[35]);
  fprintf(gbl.dbgfil, " %seqvchk", _TNO(currdir->x[19] & 0x1));
  fprintf(gbl.dbgfil,
          "   %slstval,%ssplit,%svintr,%spipei,%sdualopi,%sbounds,%ssse\n",
          _TNO(currdir->x[19] & 0x2), _FNO(currdir->x[19] & 0x4),
          _TNO(currdir->x[34] & 0x8), _FNO(currdir->x[4] & 0x1),
          _FNO(currdir->x[4] & 0x2), _FNO(currdir->x[70] & 0x2),
          _TNO(currdir->x[19] & 0x400));
  fprintf(gbl.dbgfil, "   altcode: vector=%d,swpipe=%d,unroll=%d\n",
          currdir->x[16], currdir->x[17], currdir->x[18]);
  fprintf(gbl.dbgfil, "   %sfunc32, %sframe", _FNO(currdir->x[119] & 0x4),
          _TNO(currdir->x[121] & 0x1));
  fprintf(gbl.dbgfil, " info=%0x", currdir->x[0]);
  fprintf(gbl.dbgfil, "   stripsize:%d", currdir->x[38]);
  if (currdir->x[34] & 0x100000)
    fprintf(gbl.dbgfil, "   nolastdim");
  if (currdir->x[34] & 0x800)
    fprintf(gbl.dbgfil, "   safe_last_val");
  fprintf(gbl.dbgfil, "\n");
  fprintf(gbl.dbgfil, "   %sconcur,%sinvarif,%sunroll=c,%sunroll=n,",
          _TNO(currdir->x[34] & (0x20 | 0x10)), _TNO(currdir->x[19] & 0x80),
          _TNO(currdir->x[11] & 0x1), _TNO(currdir->x[11] & 0x2));
  fprintf(gbl.dbgfil, "unroll=c:%d,unroll=n:%d", currdir->x[9], currdir->x[10]);
#ifdef FE90
  fprintf(gbl.dbgfil, ",%sindependent", _FNO(currdir->x[19] & 0x100));
#endif
  fprintf(gbl.dbgfil, "\n");
}

static void
dmp_lpprg(int i)
{
  LPPRG *p;
#ifdef FE90
  NEWVAR *nv;
#endif

  p = direct.lpg.stgb + i;
  fprintf(gbl.dbgfil, "---dirset (%4d) for loop, lines %d, %d\n", i,
          p->beg_line, p->end_line);
  dmp_dirset(&p->dirset);
#ifdef FE90
  if (p->indep) {
    REDUCVAR *redp;
    REDUC_JA *redjap;
    REDUC_JA_SPEC *specp;

    fprintf(gbl.dbgfil, "   onhome ast %d\n", p->indep->onhome);

    fprintf(gbl.dbgfil, "   new variables:");
    for (nv = p->indep->new_list; nv != NULL; nv = nv->next)
      fprintf(gbl.dbgfil, " %d(%s)", nv->var, SYMNAME(nv->var));
    fprintf(gbl.dbgfil, "\n");

    fprintf(gbl.dbgfil, "   reduction variables:");
    for (redp = p->indep->reduction_list; redp; redp = redp->next)
      fprintf(gbl.dbgfil, " %d(%s)", redp->var, SYMNAME(redp->var));
    fprintf(gbl.dbgfil, "\n");

    fprintf(gbl.dbgfil, "   JAHPF reduction variables:");
    for (redjap = p->indep->reduction_ja_list; redjap; redjap = redjap->next) {
      for (specp = redjap->speclist; specp; specp = specp->next)
        fprintf(gbl.dbgfil, " %d(%s)", specp->var, SYMNAME(specp->var));
    }
    fprintf(gbl.dbgfil, "\n");

    fprintf(gbl.dbgfil, "   index variables:");
    for (nv = p->indep->index_list; nv != NULL; nv = nv->next)
      fprintf(gbl.dbgfil, " %d(%s)", nv->var, SYMNAME(nv->var));
    fprintf(gbl.dbgfil, "\n");
  }
  if (p->index_reuse_list) {
    INDEX_REUSE *irp;

    fprintf(gbl.dbgfil, "   JAHPF INDEX_REUSE variables:");
    for (irp = p->index_reuse_list; irp; irp = irp->next) {
      for (nv = irp->reuse_list; nv; nv = nv->next)
        fprintf(gbl.dbgfil, " %d(%s)", nv->var, SYMNAME(nv->var));
    }
    fprintf(gbl.dbgfil, "\n");
  }
#endif
}
#endif

#ifdef FE90

void
direct_export(FILE *ff)
{
  int i;
  SVDIR *p;

  fprintf(ff, "A:%d\n", svdir.avail);
  fprintf(ff, "rou: --------------------\n");
  wr_dir(ff, &svdir.stgb[0]);
  for (i = 1; i < svdir.avail; i++) {
    p = svdir.stgb + i;
    fprintf(ff, "%d: --------------------\n", i);
    fprintf(ff, "b:%d e:%d\n", p->beg_line, p->end_line);
    wr_dir(ff, p);
  }
}

static void
update_rou_begin(void)
{
  SVDIR *df;
  DIRSET *new, *old, *older;
  int i;

  df = &svdir.stgb[0];
  new = &direct.rou_begin;
  old = &direct.gbl;
  older = &inigbl;

  diff_dir(df, new, old);

  if (old->opt != older->opt)
    df->change.opt = 1;
  if (old->vect ^ older->vect)
    df->change.vect = 1;
  if (old->depchk != older->depchk)
    df->change.depchk = 1;
  for (i = 0; i < sizeof(flg.x) / sizeof(int); i++) {
    if (is_xflag_bit(i)) {
      if (old->x[i] ^ older->x[i])
        df->change.x[i] = 1;
    } else {
      if (old->x[i] != older->x[i])
        df->change.x[i] = 1;
    }
  }
  svdir.avail = direct.lpg.avail;
}

static void
diff_dir(SVDIR *df, DIRSET *new, DIRSET *old)
{
  int i;

  df->newset = *new;

  df->change.opt = new->opt != old->opt;
  df->change.vect = new->vect ^ old->vect;
  df->change.depchk = new->depchk != old->depchk;
  for (i = 0; i < sizeof(flg.x) / sizeof(int); i++) {
    if (is_xflag_bit(i))
      df->change.x[i] = new->x[i] ^ old->x[i];
    else
      df->change.x[i] = new->x[i] != old->x[i];
  }
}

static void
wr_dir(FILE *ff, SVDIR *dd)
{
  int i;

  if (dd->change.opt)
    fprintf(ff, "o:%x %x\n", dd->change.opt, dd->newset.opt);
  if (dd->change.vect)
    fprintf(ff, "v:%x %x\n", dd->change.vect, dd->newset.vect);
  if (dd->change.depchk)
    fprintf(ff, "d:%x %x\n", dd->change.depchk, dd->newset.depchk);
  for (i = 0; i < sizeof(flg.x) / sizeof(int); i++)
    if (dd->change.x[i])
      fprintf(ff, "x%d:%x %x\n", i, dd->change.x[i], dd->newset.x[i]);
  fprintf(ff, "z\n");
}
#endif

static FILE *dirfil;
static int ilmlinenum = 0;
#define MAXLINELEN 4096
static char line[MAXLINELEN];
static int read_line(void);
static int rd_dir(DIRSET *);

int
direct_import(FILE *ff)
{
  int ret;
  int i;
  int idx;
  LPPRG *lpprg;

  ilmlinenum = 0;
  dirfil = ff;

  /* read size of the lpg table */
  if (read_line())
    goto err;
  ret = sscanf(line, "A:%d", &direct.lpg.avail);
  if (ret != 1)
    goto err;
  NEEDB(direct.lpg.avail, direct.lpg.stgb, LPPRG, direct.lpg.size,
        direct.lpg.avail + 8);

  /* read routine directives */
  if (read_line())
    goto err; /* rou: line */
  if (line[0] != 'r')
    goto err;
  direct.rou_begin = direct.gbl;
  if (rd_dir(&direct.rou_begin))
    goto err;

  /* read the loop directives */
  for (i = 1; i < direct.lpg.avail; i++) {
    lpprg = direct.lpg.stgb + i;

    if (read_line())
      goto err; /* idx: line */
    ret = sscanf(line, "%d: ", &idx);
    if (ret != 1)
      goto err;
    if (i != idx)
      goto err;

    if (read_line())
      goto err; /* b:lineno e:lineno */
    ret = sscanf(line, "b:%d e:%d", &lpprg->beg_line, &lpprg->end_line);
    if (ret != 2)
      goto err;

    lpprg->dirset = direct.rou_begin;
    if (rd_dir(&lpprg->dirset))
      goto err;
#if DEBUG
    if (DBGBIT(1, 512))
      dmp_lpprg(i);
#endif
  }

  return ilmlinenum;
err:
  printf("DIRECTIVES error\n");
  return ilmlinenum;
}

static int
read_line(void)
{
  char *ret;
  ret = fgets(line, MAXLINELEN - 1, dirfil);
  ++ilmlinenum;
  if (ret == NULL)
    return 1;
  return 0;
} /* read_line */

static int
rd_dir(DIRSET *dd)
{
  int ret;
  int v;
  int change;
  int idx;

#undef ST_VAL
#undef ST_BV
#undef UADDR
#define ST_VAL(m) \
  if (change)     \
  dd->m = v
#define ST_BV(m) \
  if (change)    \
  dd->m = (v & change) | (dd->m & ~change)
#define UADDR(x) (unsigned int *) & x

  while (true) {
    /* read input line */
    if (read_line())
      return 1;
    switch (line[0]) {
    case 'z':
      return 0;
    case 'o': /* read opt line */
      ret = sscanf(line, "o:%x %x", UADDR(change), UADDR(v));
      if (ret != 2)
        return 1;
      {
        ST_VAL(opt);
      }
      break;
    case 'v': /* read vect line */
      ret = sscanf(line, "v:%x %x", UADDR(change), UADDR(v));
      if (ret != 2)
        return 1;
      if (dd != &direct.rou_begin) {
        ST_BV(vect);
      } else {
        ST_VAL(vect);
      }
      break;
    case 'd': /* read depchk line */
      ret = sscanf(line, "d:%x %x", UADDR(change), UADDR(v));
      if (ret != 2)
        return 1;
      {
        ST_VAL(depchk);
      }
      break;
    case 'x':
      /* read x flag.  The line is of the form:
       *   x<n>:change new [idx:change new ]...
       * <n> is in decimal; change & new are in hex.
       */
      ret = sscanf(line, "x%d:%x %x", &idx, UADDR(change), UADDR(v));
      if (ret != 3)
        return 1;
      if (dd == &direct.rou_begin) {
        ST_VAL(x[idx]);
      } else if (is_xflag_bit(idx)) {
        ST_BV(x[idx]);
      } else {
        ST_VAL(x[idx]);
      }
      break;
    default:
      return 1;
    }
  }
  return 0;
}
