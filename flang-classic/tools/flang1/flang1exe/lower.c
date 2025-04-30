/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief lower STD/AST to back-end ILM structure
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "dtypeutl.h"
#include "ast.h"
#include "fdirect.h"
#include "rtlRtns.h"

#define INSIDE_LOWER
#include "lower.h"

static int errors;

static void lower_start_subroutine(int);
static void lower_end_subroutine(void);
static void lower_program(int);
static void save_contained(void);
static void init_contained(void);
static void lower_directives(void);

static int last_contained = 0, size_contained = 0;
static int *contained = NULL;

static int last_contained_char = 0, size_contained_char = 0;
static char *contained_char = NULL;

static int *outerflags = NULL;

#define STB_LOWER() ((gbl.outfil == lowersym.lowerfile) && gbl.stbfil)
#ifdef FLANG_LOWER_UNUSED
static void lower_directives_llvm(void);
#endif

#if DEBUG
void
LowerTraceOutput(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);
  if (DBGBIT(47, 2)) {
    if (gbl.dbgfil) {
      vfprintf(gbl.dbgfil, fmt, argptr);
      fprintf(gbl.dbgfil, "\n");
    } else {
      fprintf(stderr, "Trace: ");
      vfprintf(stderr, fmt, argptr);
      fprintf(stderr, "\n");
    }
    va_end(argptr);
  }
} /* LowerTraceOutput */
#endif

void
lerror(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);
  if (gbl.dbgfil && gbl.dbgfil != stderr) {
    fprintf(gbl.dbgfil, "Lowering Error: ");
    vfprintf(gbl.dbgfil, fmt, argptr);
    fprintf(gbl.dbgfil, "\n");
    va_start(argptr, fmt);
  }
  fprintf(stderr, "Lowering Error: ");
  vfprintf(stderr, fmt, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
  ++errors;
} /* lerror */

#if DEBUG
int
lower_ndtypeg(int ast)
{
  /* If execution gets here, then the macro NDTYPEG has been redefined.  Use
   * the underlying AST field name (w19) to prevent recursion */
  if (astb.stg_base[ast].w19 < 0) {
    ast_error("NDTYPE not set", ast);
    return A_DTYPEG(ast);
  }
  return astb.stg_base[ast].w19;
} /* lower_ndtypeg */
#endif

/** \brief Lower the STDs and ASTs to something that the back end can use.

    Essentially, lower to ILMs, save them to a file.  In the mean time, save
    the symbol table to the same file, as well as anything else needed to
    compile the program.
 */
void
lower(int staticinit)
{
  int std, nextstd, a;
  int did_debug_label;
  FILE *save_lowerfile;
  int save_internal, save_docount, save_outersub, save_outerentries;

  /* clear the VISIT flags for all symbols */
  lower_clear_visit_fields();

  /* set 'noconflict' bit, pointer types */
  lower_set_symbols();

#if DEBUG
  if (DBGBIT(47, 4)) {
    symdmp(gbl.dbgfil, 0);
    dmp_dtype();
  }
#endif

  if (staticinit) {
    /* static init routine for host subprogram */
    save_docount = lowersym.docount;
    save_lowerfile = lowersym.lowerfile;
    save_outersub = lowersym.outersub;
    save_outerentries = lowersym.outerentries;
    save_internal = gbl.internal;
    lowersym.lowerfile = gbl.outfil;
    lowersym.docount = 0;
    lowersym.outersub = 0;
    lowersym.outerentries = 0;
    gbl.internal = 0;
  } else {
    switch (gbl.internal) {
    case 0:
      /* an outer subprogram that does not contain any others */
      /* no need to copy to/from a temp file, copy directly to output file */
      lowersym.lowerfile = gbl.outfil;
      lowersym.docount = 0;
      break;
    case 1:
      /* an outer subprogram that contains others */
      /* create a temporary file */
      lowersym.lowerfile = tmpfile();
      if (lowersym.lowerfile == NULL) {
        error(0, 4, 0, "could not open temporary ILM symbol file", "");
      }
      init_contained();
      lowersym.docount = 0;
      break;
    default:
      /* a contained subprogram; leave lowersym.lowerfile at the temp file */
      /* save all the entry point names in a list of contained names */
      if (lowersym.lowerfile == NULL || lowersym.lowerfile == gbl.outfil ||
          contained == NULL) {
        /* must have been an error in the containing routine; give up */
        return;
      }
      save_contained();
      break;
    }
  }

  errors = 0;

  lower_line = FUNCLINEG(gbl.currsub);

  lower_init_sym();

  lower_fill_member_parent();

  if (gbl.rutype != RU_BDATA) {
    lower_add_pghpf_commons();
  }
  /* mark all common blocks */
  lower_linearized();
  lower_common_sizes();
  lower_namelist_plists();

  /* clear the A_OPT1 and A_OPT2 fields for use here */
  for (a = 0; a < astb.stg_avail; ++a) {
#if DEBUG
    A_NDTYPEP(a, -1);
#endif
    A_ILMP(a, 0);
    A_BASEP(a, 0);
  }

  /* put out header lines */
  lower_ilm_header();

  /* should this be converted to a subroutine? */
  if (gbl.rutype == RU_FUNC) {
    switch (DTY(DTYPEG(gbl.currsub))) {
    case TY_DERIVED:
    case TY_STRUCT:
      if (!CFUNCG(gbl.currsub)) {
        gbl.rutype = RU_SUBR;
      }
      break;
    }
  }

  lower_start_subroutine(gbl.rutype);

  lower_data_stmts();

  /* generate the code for each AST tree */
  did_debug_label = 0;
  for (std = STD_NEXT(0); std; std = nextstd) {
    nextstd = STD_NEXT(std);
    if (did_debug_label == 0 &&
        (staticinit || STD_LINENO(std) || nextstd == 0 || STD_ORIG(std))) {
      if (XBIT(52, 4) || XBIT(58, 0x22)) {
        fill_entry_bounds(gbl.currsub, STD_LINENO(std));
      }
      lower_debug_label();
      did_debug_label = 1;
    }
    lower_stmt(std, STD_AST(std), STD_LINENO(std), STD_LABEL(std));
  } /* for std */

  /* mark the current subroutine */
  lower_mark_entries();

  lower_set_craypointer();

  lower_check_generics();

  /* need to process iface here to lower correct data types and iface symbols */
  stb_fixup_llvmiface();

  lower_sym_header();

  lower_fileinfo();

  lower_program(gbl.rutype);

  if (!staticinit && gbl.internal >= 1) {
    /* put out all outer symbols */
    lower_outer_symbols();
  }

  /* dump out the data types */
  lower_data_types();

  /* dump out the used symbols */
  lower_symbols();
  lower_end_subroutine();

  lower_ilm_finish();
  lower_finish_sym();
  lower_exp_finish();
  lower_directives();
  if (errors) {
    interr("Errors in Lowering", errors, 4);
  }
#if DEBUG
  if (DBGBIT(47, 4)) {
    dump_std();
    symdmp(gbl.dbgfil, 0);
    dmp_dtype();
  }
#endif
  for (a = 0; a < astb.stg_avail; ++a) {
    A_NDTYPEP(a, 0);
  }
  lower_unset_symbols();
  if (staticinit) {
    /* static init routine for host subprogram */
    lowersym.lowerfile = save_lowerfile;
    lowersym.docount = save_docount;
    gbl.internal = save_internal;
    lowersym.outersub = save_outersub;
    lowersym.outerentries = save_outerentries;
  } else if (gbl.internal == 0) {
    lowersym.outersub = 0;
    lowersym.outerentries = 0;
  } else if (gbl.internal == 1) {
    lowersym.outersub = gbl.currsub;
    lowersym.outerentries = gbl.entries;
    lowersym.last_outer_sym = stb.stg_avail;
  }
} /* lower */

/** \brief Clear the A_OPT1 and A_OPT2 fields for a single ast, using
 * ast_traverse.
 */
void
lower_clear_opt(int ast, int *unused)
{
  A_ILMP(ast, 0);
  A_BASEP(ast, 0);
} /* lower_clear_opt */

void
lower_debug_label(void)
{
  plower("olnn", "BOS", 0, gbl.findex, 0);
  plower("o", "ENLAB");
  plower("o", "--------------------");
} /* lower_debug_label */

static void
lower_start_subroutine(int rutype)
{
  /* int lab1, lab2; */
  int syminit, ilm;

  lowersym.labelcount = -1;
  lowersym.last_lineno = -1;

  plower("olnn", "BOS", gbl.funcline, gbl.findex, 0);
  plower("o", "NOP");
  plower("o", "--------------------");

  switch (rutype) {
  case RU_BDATA:
    break;
  case RU_SUBR:
  case RU_FUNC:
    /* generate code to fill bounds of adjustable arrays */
    lower_start_stmt(gbl.funcline, 0, FALSE, 0);
    lower_pointer_init();
    lower_end_stmt(0);
    break;
  case RU_PROG:
    /* call 'init' */
    lower_start_stmt(gbl.funcline, 0, FALSE, 0);
    if (1 /* flg.defaulthpf || XBIT(49,0x1000)*/) {
      syminit = lower_makefunc(mkRteRtnNm(RTE_init), DT_NONE, TRUE);
      ilm = plower("oS", "ICON", lowersym.intzero);
      plower("onsiC", "CALL", 1, syminit, ilm, syminit);
    }
    lower_pointer_init();
    lower_end_stmt(0);
    break;
  }
} /* lower_start_subroutine */

static void
lower_end_subroutine(void)
{
  fprintf(lowersym.lowerfile, "end\n");
  if (STB_LOWER()) {
    fprintf(gbl.stbfil, "end\n");
  }
  uncouple_callee_args();
  ccff_lower(lowersym.lowerfile);
} /* lower_end_subroutine */

static void
init_contained(void)
{
  int sz;
  lowersym.first_outer_sym = stb.firstusym;
  lowersym.last_outer_sym = stb.stg_avail;
  lowersym.last_outer_sym_orig = stb.stg_avail;
  last_contained = 0;
  size_contained = 100;
  NEW(contained, int, size_contained);
  BZERO(contained, int, size_contained);

  last_contained_char = 0;
  size_contained_char = 1000;
  NEW(contained_char, char, size_contained_char);

  sz = lowersym.last_outer_sym_orig - lowersym.first_outer_sym;
  NEW(outerflags, int, sz);
  BZERO(outerflags, int, sz);
} /* init_contained */

static void
save_contained_name(char *name)
{
  int len;

  NEED(last_contained + 1, contained, int, size_contained,
       size_contained + 100);
  contained[last_contained] = last_contained_char;
  ++last_contained;

  len = strlen(name);
  NEED(last_contained_char + len + 1, contained_char, char, size_contained_char,
       size_contained_char + 1000);
  strcpy(contained_char + last_contained_char, name);
  last_contained_char += len + 1;
} /* save_contained_name */

static void
markid(int astx, int *unused)
{
  if (A_TYPEG(astx) == A_ID) {
    int sptr, dtype;
    sptr = A_SPTRG(astx);
    if (sptr >= lowersym.first_outer_sym && sptr < lowersym.last_outer_sym_orig)
      outerflags[sptr - lowersym.first_outer_sym] |= 1;
    switch (STYPEG(sptr)) {
    case ST_ARRAY:
    case ST_VAR:
    case ST_UNION:
    case ST_STRUCT:
      if (MIDNUMG(sptr) && MIDNUMG(sptr) >= lowersym.first_outer_sym &&
          MIDNUMG(sptr) < lowersym.last_outer_sym_orig)
        outerflags[MIDNUMG(sptr) - lowersym.first_outer_sym] |= 1;
      if (SDSCG(sptr) && SDSCG(sptr) >= lowersym.first_outer_sym &&
          SDSCG(sptr) < lowersym.last_outer_sym_orig)
        outerflags[SDSCG(sptr) - lowersym.first_outer_sym] |= 1;
      break;
    default:
      break;
    }
    dtype = DTYPEG(sptr);
    if (dtype && DTY(dtype) == TY_ARRAY) {
      int numdim, i;
      numdim = ADD_NUMDIM(dtype);
      for (i = 0; i < numdim; ++i) {
        if (ADD_LWBD(dtype, i) != 0)
          ast_traverse_more(ADD_LWBD(dtype, i), unused);
        if (ADD_UPBD(dtype, i) != 0)
          ast_traverse_more(ADD_UPBD(dtype, i), unused);
        if (ADD_LWAST(dtype, i) != 0)
          ast_traverse_more(ADD_LWAST(dtype, i), unused);
        if (ADD_UPAST(dtype, i) != 0)
          ast_traverse_more(ADD_UPAST(dtype, i), unused);
        if (ADD_MLPYR(dtype, i) != 0)
          ast_traverse_more(ADD_MLPYR(dtype, i), unused);
        if (ADD_EXTNTAST(dtype, i) != 0)
          ast_traverse_more(ADD_EXTNTAST(dtype, i), unused);
      }
      if (ADD_ZBASE(dtype) != 0)
        ast_traverse_more(ADD_ZBASE(dtype), unused);
      if (ADD_NUMELM(dtype) != 0)
        ast_traverse_more(ADD_NUMELM(dtype), unused);
    }
  }
} /* markid */

static void
save_contained(void)
{
  int sptr, stdx;
  for (sptr = stb.firstusym; sptr < stb.stg_avail; ++sptr) {
    if (STYPEG(sptr) == ST_ENTRY) {
      save_contained_name(SYMNAME(sptr));
    }
  }
  ast_visit(1, 1);
  for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
    ast_traverse(STD_AST(stdx), NULL, markid, NULL);
  }
  ast_unvisit_norepl();
} /* save_contained */

void
lower_end_contains(void)
{
#define LOWERBUFSIZ 10000
  char buffer[LOWERBUFSIZ];
  int symbolslist;
  int outer;
  LOGICAL stbok = TRUE;

  Trace(("end of containing routine"));
  if (lowersym.lowerfile == NULL) {
    Trace(("no lowersym.lowerfile"));
    if (contained)
      FREE(contained);
    if (contained_char)
      FREE(contained_char);
    if (outerflags)
      FREE(outerflags);
    return;
  }
  if (lowersym.lowerfile == gbl.outfil) {
    /* should not happen */
    Trace(("lowersym.lower is gbl.outfil"));
    if (contained)
      FREE(contained);
    if (contained_char)
      FREE(contained_char);
    if (outerflags)
      FREE(outerflags);
    return;
  }

  rewind(lowersym.lowerfile);
  symbolslist = 1; /* 1 => reading symbols */
  outer = 1;
  while (fgets(buffer, LOWERBUFSIZ, lowersym.lowerfile) != NULL) {

    if (buffer[0] == 'e') {
      switch (symbolslist) {
        int c;
      case 1: /* symbols */
        /* put out names of all the contained subroutines. */
        for (c = 0; c < last_contained; ++c) {
          char *ch;
          ch = contained_char + contained[c];
          fprintf(gbl.outfil, "contained %lu:%s\n", strlen(ch), ch);
          if (gbl.stbfil)
            fprintf(gbl.stbfil, "contained %lu:%s\n", strlen(ch), ch);

        }
        if (outer) {
          /* put out the names of outer block symbols referenced
           * in the internal routines */
          int sptr;
          for (sptr = lowersym.first_outer_sym; sptr < lowersym.last_outer_sym;
               ++sptr) {
            switch (STYPEG(sptr)) {
            case ST_ARRAY:
            case ST_DESCRIPTOR:
            case ST_VAR:
            case ST_UNION:
            case ST_STRUCT:
            case ST_PLIST:
              if (sptr >= lowersym.last_outer_sym_orig ||
                  (outerflags[sptr - lowersym.first_outer_sym] &
                   1) /*REFINTG(sptr)*/) {
#if DEBUG
                if (DBGBIT(47, 8)) {
                  if (GSCOPEG(sptr) || XBIT(7, 0x200000))
                  {
                    fprintf(gbl.outfil, "global:%s\n", getprint(sptr));
                    if (gbl.stbfil)
                      fprintf(gbl.stbfil, "global:%s\n", getprint(sptr));
                  }
                } else
#endif
                    if (GSCOPEG(sptr) || XBIT(7, 0x200000))
                {
                  fprintf(gbl.outfil, "global:%d\n", sptr);
                  if (gbl.stbfil)
                    fprintf(gbl.stbfil, "global:%d\n", sptr);
                }
              }
              break;
            case ST_PROC:
              if (SCG(sptr) == SC_DUMMY &&
                  (sptr >= lowersym.last_outer_sym_orig ||
                   (outerflags[sptr - lowersym.first_outer_sym] & 1))) {
#if DEBUG
                if (DBGBIT(47, 8)) {
                  fprintf(gbl.outfil, "global:%s\n", getprint(sptr));
                  if (gbl.stbfil)
                    fprintf(gbl.stbfil, "global:%s\n", getprint(sptr));
                } else
#endif
                {
                  fprintf(gbl.outfil, "global:%d\n", sptr);
                  if (gbl.stbfil)
                    fprintf(gbl.stbfil, "global:%d\n", sptr);
                }
              }
              break;
            default:
              break;
            }
          }
          outer = 0;
        }
        break;
      case 2: /* ilms */
        break;
      case 3: /* directives */
        symbolslist = 0;
        break;
      }
      /* step theu symbolslist, ilmlist, directives list */
      symbolslist++;
    }
    fputs(buffer, gbl.outfil);
    if (gbl.stbfil) {
      /* Skip ast */
      if (buffer[0] == 'A') { /* AST2ILM */
        stbok = FALSE;
      } else if (!stbok && buffer[0] == 'T') { /* TOILM */
        stbok = TRUE;
      }
      if (stbok)
        fputs(buffer, gbl.stbfil);

      if (buffer[0] == 'e') { /* end */
        stbok = FALSE;
      }
    }
  }
  fclose(lowersym.lowerfile);
  if (gbl.currmod)
    lowersym.lowerfile = gbl.outfil;
  else
    lowersym.lowerfile = NULL;
  FREE(contained);
  FREE(contained_char);
  FREE(outerflags);
} /* lower_end_contains */

static void
lower_program(int rutype)
{
  const char *s;
  switch (rutype) {
  case RU_SUBR:
    s = "Subroutine";
    break;
  case RU_FUNC:
    s = "Function";
    break;
  case RU_PROG:
    s = "Program";
    break;
  case RU_BDATA:
    s = "Blockdata";
    break;
  }

  fprintf(lowersym.lowerfile, "procedure:%s\n", s);
  if (STB_LOWER())
    fprintf(gbl.stbfil, "procedure:%s\n", s);
} /* lower_program */

static void
lower_directives(void)
{
  /*lowersym.lowerfile*/
  fprintf(lowersym.lowerfile, "DIRECTIVES version %d/%d\n", VersionMajor,
          VersionMinor);
  direct_export(lowersym.lowerfile);
  fprintf(lowersym.lowerfile, "end\n");

/* Don't think we need to lower directives */
} /* lower_directives */

/** \brief Notify back end of the end of a module.
 */
void
lower_constructor(void)
{
  fprintf(gbl.outfil, "CONSTRUCTORACC\n");
} /* lower_constructor */

/*
 * Routines and data for determining how to return bind(C) function retvals
 * according to the ABI
 * These values must be kept insync with the values in the BE file exp_rte.c
 */

#if defined(TARGET_WIN_X8664)
#define MAX_PASS_STRUCT_SIZE 8
#else
#define MAX_PASS_STRUCT_SIZE 16
#endif

#define PACK(i, j) (((j) << 8) | ((i)&0xFF))
#define UNPACKLOW(i) ((i)&0xFF)
#define UNPACKHIGH(i) ((i) >> 8)
#define REGSIZE 8

#define CLASS_INT(t) (t >= CLASS_INT1 && t <= CLASS_INT8)
#define CLASS_SSESP(t) (t == CLASS_SSESP4 || t == CLASS_SSESP8)
#define CLASS_SSE(t)                                             \
  (t == CLASS_SSESP4 || t == CLASS_SSESP8 || t == CLASS_SSEDP || \
   t == CLASS_SSEQ)

static int regclass[2];

/* return the sum of a and b using merge rules (integer over float) */
static int
addclasses(int a, int b)
{
  int class;

  if (a == b) {
    if (CLASS_INT(a)) {
      class = a + b;
      if (class > CLASS_INT8)
        class = CLASS_MEM;
    } else if (a == CLASS_SSESP4)
      class = CLASS_SSESP8;
    else
      class = CLASS_MEM;
  } else if (a == CLASS_NONE)
    class = b;
  else if (b == CLASS_NONE)
    class = a;
  else if (a == CLASS_MEM || b == CLASS_MEM)
    class = CLASS_MEM;
  else if (CLASS_INT(a) && CLASS_INT(b)) {
    class = a + b;
    if (class > CLASS_INT8)
      class = CLASS_MEM;
  } else if (CLASS_INT(a) && b == CLASS_SSESP4) {
    class = a + CLASS_INT4;
    if (class > CLASS_INT8)
      class = CLASS_MEM;
  } else if (CLASS_INT(b) && a == CLASS_SSESP4) {
    class = b + CLASS_INT4;
    if (class > CLASS_INT8)
      class = CLASS_MEM;
  } else if (a == CLASS_SSESP4 && b == CLASS_SSESP4)
    class = CLASS_SSESP8;
  else
    class = CLASS_MEM;

  return class;
}

static void
trav_struct(int dtype, int off)
{
  int i;
  int d;
  int addr, mem;
  int elems, elemsize;
  int regi;
  int tmpclass[2];
  ADSC *ad;

#if DEBUG
  assert(off < REGSIZE * 2, "trav_struct - bad offset", off, 3);
#endif

  regi = off / REGSIZE; /* are we looking at 1st or 2nd reg. */
  tmpclass[0] = CLASS_NONE;
  tmpclass[1] = CLASS_NONE;

  switch (DTY(dtype)) {
  case TY_CHAR:
  case TY_BINT:
  case TY_BLOG:
    regclass[regi] = CLASS_INT1;
    return;
  case TY_SINT:
  case TY_SLOG:
    regclass[regi] = CLASS_INT2;
    return;
  case TY_INT:
  case TY_LOG:
    regclass[regi] = CLASS_INT4;
    return;
  case TY_INT8:
  case TY_PTR:
  case TY_LOG8:
    regclass[regi] = CLASS_INT8;
    return;
  case TY_128: /*m128*/
    regclass[regi] = CLASS_SSEQ;
    return;
  case TY_QUAD:
#if DEBUG
    if (sizeof(DT_QUAD) == 16)
      interr("trav_struct: update handling of long doubles", dtype, 3);
#endif
    /* We're treating this like DBLE for now. */
    FLANG_FALLTHROUGH;
  case TY_DBLE:
    regclass[regi] = CLASS_SSEDP;
    return;
  case TY_FLOAT:
    regclass[regi] = CLASS_SSESP4;
    return;
  case TY_CMPLX:
    regclass[regi] = CLASS_SSESP8;
    return;
  case TY_DCMPLX:
    assert(regi == 0, "trav_struct - bad offset for DCMPLX", off, 3);
    regclass[0] = CLASS_SSEDP;
    regclass[1] = CLASS_SSEDP;
    return;
  case TY_STRUCT:
  case TY_DERIVED:
    /* regclass will be the sum of the members in its eightbyte */
    for (mem = DTY(dtype + 1); mem != NOSYM; mem = SYMLKG(mem)) {
      if (CLASSG(mem) && VTABLEG(mem) && (TBPLNKG(mem) || FINALG(mem)))
        continue;
      addr = ADDRESSG(mem) + off;
      trav_struct(DTYPEG(mem), addr);
      regi = addr / REGSIZE;
      regclass[regi] = addclasses(regclass[regi], tmpclass[regi]);
      tmpclass[0] = regclass[0];
      tmpclass[1] = regclass[1];
    }
    return;

  case TY_ARRAY:
    /* handle case of array in struct/union */
    ad = AD_DPTR(dtype);
    d = AD_NUMELM(ad);
    if (A_TYPEG(d) != A_CNST)
      return;
    elems = ad_val_of(A_SPTRG(d));
    elemsize = size_of(DTY(dtype + 1));
    if (elems > 0 && elems <= 16) {
      for (i = 0; i < elems; i++) {
        addr = elemsize * i + off;
        trav_struct(DTY(dtype + 1), addr);
        regi = addr / REGSIZE;
        regclass[regi] = addclasses(regclass[regi], tmpclass[regi]);
        tmpclass[0] = regclass[0];
        tmpclass[1] = regclass[1];
      }
    } else {
#if DEBUG
      assert(FALSE, "trav_struct - unexpected elems", elems, 3);
#endif
      ;
    }
    return;

  default:
    interr("invalid type for trav_struct", dtype, 3);
    return;
  }
}

#if defined(TARGET_WIN_X8664)
/* all small structs are passed and returned as if they were integers */
static void
adjust_regclass(void)
{
  int i;

  for (i = 0; i < 2; ++i) {
    switch (regclass[i]) {
    case CLASS_SSESP4:
      regclass[i] = CLASS_INT4;
      break;
    case CLASS_SSESP8:
    case CLASS_SSEDP:
    case CLASS_SSEQ: /*m128 TBD*/
      regclass[i] = CLASS_INT8;
      break;
    default:
      break;
    }
  }
}
#endif

static int
pass_struct(int dtype)
{
  int retval;

  /* initialize regclass */
  regclass[0] = CLASS_NONE;
  regclass[1] = CLASS_NONE;

#if DEBUG
  assert(DTY(dtype) == TY_STRUCT || DTY(dtype) == TY_DERIVED ||
             DTY(dtype) == TY_UNION || DT_ISCMPLX(dtype),
         "pass_struct - unexpected type", dtype, 2);
#endif

  /* trav_struct will change regclass in traversal */
  trav_struct(dtype, 0);

#if DEBUG
  assert(regclass[0] != CLASS_NONE, "pass_struct - bad retval", 0, 3);
#endif

#if defined(TARGET_WIN_X8664)
  adjust_regclass();
#endif

  if (regclass[0] == CLASS_MEM || regclass[1] == CLASS_MEM)
    retval = CLASS_MEM;
  else
    retval = PACK(regclass[0], regclass[1]);

  return retval;
}

static int
check_struct(int dtype)
{
  int size;

  size = size_of(dtype);
#if defined(TARGET_WIN_X8664)
  switch (size) {
  case 1:
    return CLASS_INT1;
  case 2:
    return CLASS_INT2;
  case 4:
    return CLASS_INT4;
  case 8:
    return CLASS_INT8;
  default:
    break;
  }
  return CLASS_PTR;
#else
  if (size <= MAX_PASS_STRUCT_SIZE)
    return pass_struct(dtype);
  else
    return CLASS_MEM;
#endif
}

int
check_return(int retdtype)
{
  if (DTY(retdtype) == TY_DERIVED || DTY(retdtype) == TY_STRUCT ||
      DTY(retdtype) == TY_UNION || DT_ISCMPLX(retdtype))
    return check_struct(retdtype);
  else
    return CLASS_INT4; /* something not CLASS_MEM */
}

#ifdef FLANG_LOWER_UNUSED
static void
lower_directives_llvm(void)
{
  int i, n;

  if (!STB_LOWER())
    return;

  fprintf(gbl.stbfil, "DIRECTIVES version %d/%d\n", VersionMajor, VersionMinor);
  direct_export(gbl.stbfil);
  for (n = 0; n < TPNVERSION && flg.tpvalue[n] != 0; ++n)
    ;
  if (n > 0) {
    fprintf(gbl.stbfil, "vsn %d", n);
    for (i = 0; i < n; ++i) {
      fprintf(gbl.stbfil, " %d", flg.tpvalue[i]);
    }
    fprintf(gbl.stbfil, "\n");
  }
  fprintf(gbl.stbfil, "end\n");
} /* lower_directives_llvm */
#endif

