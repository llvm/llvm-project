/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief Routines used by lower.c for lowering to ILMs.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "ast.h"
#include "semant.h"
#include "dinit.h"
#include "soc.h"
#include "gramtk.h"
#include "pragma.h"
#include "mp.h"
#include "rte.h"
#include "rtlRtns.h"

#define INSIDE_LOWER
#include "lower.h"
#include "mach.h"

static FILE *lower_ilm_file = NULL;
int lower_line;
int lower_disable_ptr_chk = 0;
int lower_disable_subscr_chk = 0;
struct lsymlists_s lsymlists;

#define VarBase 0
#define SourceBase 1
#define TargetBase 2
#define ArgumentBase 3
int get_byval(int, int);

extern void setrefsymbol(int symbol);

/*
 * define the macro, ALIGNED_ALLOC, if we're to pass an align value
 * to the alloc routines.  IF defined, the '04' versions of the alloc
 * run-time routines are called.
 */
#undef ALIGNED_ALLOC
#define ALIGNED_ALLOC

/*
 * define the macro, USE_LARGE_SIZE, if we're to pass the number
 * of elements to the alloc '04' routines as a 64-bit value.
 */
#undef USE_LARGE_SIZE
#define USE_LARGE_SIZE

void
lower_ilm_header(void)
{
  /* open the output file */
  lower_ilm_file = tmpfile();
  if (lower_ilm_file == NULL) {
    error(0, 4, 0, "could not open temporary ILM file", "");
  }
  fprintf(lower_ilm_file, "AST2ILM version %d/%d\n", VersionMajor,
          VersionMinor);

} /* lower_ilm_header */

void
lower_ilm_finish(void)
{
#define LOWERBUFSIZ 10000
  char buffer[LOWERBUFSIZ];
  int nw;
  fprintf(lower_ilm_file, "end\n");
  /* append ilm file to sym file */
  nw = fseek(lower_ilm_file, 0, SEEK_SET);
  if (nw == -1)
    perror("lower_ilm_finish - fseek on lower_ilm_file");
  while (fgets(buffer, LOWERBUFSIZ, lower_ilm_file) != NULL) {
    fputs(buffer, lowersym.lowerfile);
  }
  fclose(lower_ilm_file);
  lower_ilm_file = NULL;
} /* lower_ilm_finish */

static char saveoperation[50];

static int plower_pdo(int, int);

/** \brief Print out the ILM line.
    \param fmt is a character string with the following values:
    + o - operation, must be first
    + O - same operation as most recent, must be first
    + i - ilm number
    + r - symbol number, may be zero
    + s - symbol number
    + S - symbol number, with comment containing symbol name
    + L - label symbol number, defining the label
    + C - symbol name in comment only
    + l - line number
    + d - datatype
    + n - number
    + a - argument: ilm and datatype pair, but only counts for one
    + A - argument: like 'a', but for missing optional argument
    + m - 'more', meaning don't end the ilm line here, wait for more;
          must be last, at least, anything following the m is ignored.
*/
int
plower(const char *fmt, ...)
{
  static int pcount = -1;
  static int opcount = 0;
  const char *f;
  va_list argptr;

  if (lower_ilm_file == NULL)
    return 0;
  f = fmt;
  if (f == NULL)
    return 0;
  va_start(argptr, fmt);
  if (*f == 'o' || *f == 'O') {
    char *op;
    if (*f == 'o') {
      op = va_arg(argptr, char *);
    } else {
      op = saveoperation;
    }
    if (op[0] == 'B' && op[1] == 'O' && op[2] == 'S' && op[3] == '\0') {
      pcount = -1;
    }
    opcount = ++pcount;
    fprintf(lower_ilm_file, "i%d: %s", opcount, op);
    if (op[0] == '-' && op[1] == '-' && op[2] != '-') {
      lerror("unsupported %s", op);
    }
    if (op != saveoperation)
      strcpy(saveoperation, op);
    ++f;
  }
  if (flg.debug)
    allocate_refsymbol(stb.stg_avail);

  while (*f) {
    int d;
    char chf;
    chf = *f;
    ++f; /* go to next character */
    if (chf == 'm') {
      /* stop here, more to be added */
      va_end(argptr);
      return opcount;
    } else if (chf == 'e') {
      /* end of statement, should be last */
      va_end(argptr);
      fprintf(lower_ilm_file, "\n");
      return opcount;
    }

    d = va_arg(argptr, int);
    switch (chf) {
    case 'i':
#if DEBUG
      if (DBGBIT(47, 8)) {
        fprintf(lower_ilm_file, " i-%d", opcount - d);
      } else
#endif
        fprintf(lower_ilm_file, " i%d", d);
#if DEBUG
      if (d <= 0 || d > pcount) {
        lerror("bad ilm link %d", d);
      }
#endif
      ++pcount;
      break;
    case 'r': /* may be zero */
      if (d > 0 && LOWER_SYMBOL_REPLACE(d)) {
        d = LOWER_SYMBOL_REPLACE(d);
      }
#if DEBUG
      if (DBGBIT(47, 8)) {
        if (d > 0) {
          fprintf(lower_ilm_file, " %s", getprint(d));
        } else {
          fprintf(lower_ilm_file, " s%d", d);
        }
      } else
#endif
        fprintf(lower_ilm_file, " s%d", d);
#if DEBUG
      if (d < 0 || d > stb.stg_avail) {
        lerror("bad sym link %d", d);
      }
#endif
      if (d > 0) {
        lower_visit_symbol(d);
        /* if this was a label, increment the ref count */
        if (STYPEG(d) == ST_LABEL) {
          RFCNTI(d);
        } else if (flg.debug) {
          setrefsymbol(d);
        }
      }
      ++pcount;
      break;
    case 's':
      if (d > 0 && LOWER_SYMBOL_REPLACE(d)) {
        d = LOWER_SYMBOL_REPLACE(d);
      }
#if DEBUG
      if (DBGBIT(47, 8)) {
        fprintf(lower_ilm_file, " %s", getprint(d));
      } else
#endif
        fprintf(lower_ilm_file, " s%d", d);
#if DEBUG
      if (d <= 0 || d > stb.stg_avail) {
        lerror("bad sym link %d", d);
      }
#endif
      lower_visit_symbol(d);
      /* if this was a label, increment the ref count */
      if (STYPEG(d) == ST_LABEL) {
        RFCNTI(d);
      } else if (flg.debug) {
        setrefsymbol(d);
      }
      ++pcount;
      break;
    case 'S':
      if (d > 0 && LOWER_SYMBOL_REPLACE(d)) {
        d = LOWER_SYMBOL_REPLACE(d);
      }
#if DEBUG
      if (DBGBIT(47, 8)) {
        fprintf(lower_ilm_file, " %s", getprint(d));
      } else if ((DBGBIT(47, 31) || XBIT(50, 0x10)) && *f == '\0') {
        fprintf(lower_ilm_file, " s%d	;%s", d, getprint(d));
      } else
#endif
        fprintf(lower_ilm_file, " s%d", d);
#if DEBUG
      if (d <= 0 || d > stb.stg_avail) {
        lerror("bad sym link %d", d);
      }
#endif
      lower_visit_symbol(d);
      /* if this was a label, increment the ref count */
      if (STYPEG(d) == ST_LABEL) {
        RFCNTI(d);
      } else if (flg.debug) {
        setrefsymbol(d);
      }
      ++pcount;
      break;
    case 'L':
#if DEBUG
      if (DBGBIT(47, 8)) {
        fprintf(lower_ilm_file, " %s", getprint(d));
      } else if ((DBGBIT(47, 31) || XBIT(50, 0x10)) && *f == '\0') {
        fprintf(lower_ilm_file, " s%d	;%s", d, getprint(d));
      } else
#endif
        fprintf(lower_ilm_file, " s%d", d);
#if DEBUG
      if (d <= 0 || d > stb.stg_avail) {
        lerror("bad sym link %d", d);
      }
#endif
      lower_visit_symbol(d);
      ++pcount;
      break;
    case 'C':
      if (d > 0 && LOWER_SYMBOL_REPLACE(d)) {
        d = LOWER_SYMBOL_REPLACE(d);
      }
      if (d > 0 && d < stb.stg_avail) {
#if DEBUG
        if (DBGBIT(47, 31) || XBIT(50, 0x10)) {
          fprintf(lower_ilm_file, "	;%s", getprint(d));
        }
#endif
        lower_visit_symbol(d);
      }
      /* don't increment pcount */
      break;
    case 'l':
      fprintf(lower_ilm_file, " l%d", d);
      ++pcount;
      break;
    case 'd':
    case 'D':
      if (d < 0) {
        lerror("bad datatype %d", d);
      }
      if (d == DT_ADDR) {
        if (XBIT(49, 0x100)) { /* 64-bit pointers */
          d = DT_INT8;
        } else {
          d = DT_INT;
        }
      }
#if DEBUG
      if (DBGBIT(47, 8)) {
        fprintf(lower_ilm_file, " t%d", (int)DTY(d));
      } else
#endif
        fprintf(lower_ilm_file, " t%d", d);
      ++pcount;
      if (chf == 'd')
        lower_use_datatype(d, 1);
      else
        lower_use_datatype(d, 2);
      break;
    case 'n':
      fprintf(lower_ilm_file, " n%d", d);
      ++pcount;
      break;
    case 'a':
    case 'A':
      fprintf(lower_ilm_file, " i%d", d);
#if DEBUG
      if (d <= 0 || d > pcount) {
        lerror("bad ilm link %d", d);
      }
#endif
      d = va_arg(argptr, int);
      if (d == DT_ADDR) {
        if (XBIT(49, 0x100)) { /* 64-bit pointers */
          d = DT_INT8;
        } else {
          d = DT_INT;
        }
      }
      if (chf == 'a')
        lower_use_datatype(d, 1);
      else
        lower_use_datatype(d, 2);
#if DEBUG
      if (DBGBIT(47, 8)) {
        fprintf(lower_ilm_file, " t%d", (int)DTY(d));
      } else
#endif
        fprintf(lower_ilm_file, " t%d", d);
      ++pcount;
      break;
    }
  }
  va_end(argptr);
  fprintf(lower_ilm_file, "\n");
  return opcount;
} /* plower */

int
plower_arg(const char *fmt, int arg, int dtype, int unlpoly)
{
  int ilm;
  if (!unlpoly)
    ilm = plower(fmt, "FARG", arg, dtype);
  else {
    char fff[16];
    sprintf(fff, "%sn", fmt); /* "oidn" */
    ilm = plower(fff, "FARGF", arg, dtype, 0x01);
  }
  return ilm;
}

static int
plower_pdo(int dotop, int schedtype)
{
  int opcount;
  opcount = plower("osnC", "PDO", dotop, schedtype, dotop);
  /*
   * plower() increments the RFCNT of the PDO label, but the
   * purpose of the PDO ilm is informative only.  The real
   * reference will be via the matching ENDDO, so make sure
   * the rfcnt is correct.  Otherwise, the backend, expecially
   * the vectorize, will not delete empty blocks if it's labeled
   * and the rfcnt of the label is nonzero.
   */
  RFCNTD(dotop);
  return opcount;
}

/* put out a label-list label here */
static int
plabel(int label, int value, int first)
{
  if (lower_ilm_file == NULL)
    return 0;
  ++lowersym.labelcount;
  fprintf(lower_ilm_file, "l%d label:%d value:%d first%c\n",
          lowersym.labelcount, label, value, first ? '+' : '-');
  lower_visit_symbol(label);
  RFCNTI(label);
  return lowersym.labelcount;
} /* plabel */

/** \brief Start a new block */
void
lower_start_stmt(int lineno, int label, int exec, int std)
{
  /* put out the line number information and file index info ? */
  if (STD_FINDEX(std))
    gbl.findex = STD_FINDEX(std);
  plower("olnn", "BOS", lineno, gbl.findex, 0);
#ifdef STD_TAG
  if (std && STD_TAG(std)) {
    plower("onnn", "FILE", lineno, gbl.findex, 1000 * STD_TAG(std));
  }
#endif
  /* see if there were any A_MP_CRITICAL statements just before this */
  if (std) {
    int prev;
    for (prev = STD_PREV(std); prev; prev = STD_PREV(prev)) {
      int ast = STD_AST(prev);
      switch (A_TYPEG(ast)) {
      case A_MP_CRITICAL:
        if (A_MEMG(ast))
          plower("oS", "P", A_MEMG(ast));
        else
          plower("o", "BCS");
        break;
      case A_CONTINUE:
        if (STD_LABEL(prev))
          goto exitloop;
        FLANG_FALLTHROUGH;
      case A_COMMENT:
        break;
      default:
        goto exitloop;
      }
    }
  exitloop:;
  }
  if (label) {
    plower("oL", "LABEL", label);
  }
  ast_visit(1, 1);
} /* lower_start_stmt */

void
lower_end_stmt(int std)
{
  /* see if there are any A_MP_ENDCRITICAL or A_MP_ENDPARALLEL statements
   * just after this */
  if (std) {
    int next;
    for (next = STD_NEXT(std); next; next = STD_NEXT(next)) {
      int ast = STD_AST(next);
      switch (A_TYPEG(ast)) {
      case A_MP_ENDCRITICAL:
        if (A_MEMG(ast))
          plower("oS", "V", A_MEMG(ast));
        else
          plower("o", "ECS");
        break;
      case A_COMMENT:
        break;
      default:
        goto exitloop;
      }
    }
  exitloop:;
  }
  ast_revisit(lower_clear_opt, 0);
  ast_unvisit_norepl();
  plower("o", "--------------------");
} /* lower_end_stmt */

/*
 * assign bounds variables from subscripts.
 * if subscripts >= 0, it is an array subscript descriptor index;
 * use those subscripts to fill in the lower/upper/mpyer/zbase variables.
 * if subscripts == -1, only fill in the mpyer and zbase variables,
 * using lower/upper temps.
 */
static void
put_adjarr_bounds(int dtype, int subscripts, int *pcomputednumelm)
{
  int ndim, mlpyrilm, mlpyrbaseilm, mlpyrtype, zbase, zbaseilm, i;
  int nelem; /* ast */
  ISZ_T mlpyrval, zbaseval;
  ndim = ADD_NUMDIM(dtype);
  mlpyrval = 1;
  zbaseval = 0;
  zbaseilm = 0;
  nelem = 0;

  Trace(("put bounds for array type %d, subscript descriptor %d", dtype,
         subscripts));

  mlpyrilm = 0;
  mlpyrbaseilm = 0;
  mlpyrtype = 0;
  mlpyrval = 1;

  for (i = 0; i < ndim; ++i) {
    int m, lw, lwast, up, upast, lilm, rilm, stride, strideval;
    ISZ_T lwval, upval;
    if (subscripts == -1) {
      lw = ADD_LWBD(dtype, i);
    } else {
      lw = ASD_SUBS(subscripts, i);
      if (lw && A_TYPEG(lw) == A_TRIPLE) {
        lw = A_LBDG(lw);
      } else {
        lw = astb.bnd.one;
      }
    }
    if (lw == 0) {
      lwval = 1;
    } else if (A_TYPEG(lw) == A_CNST) {
      lwval = ad_val_of(A_SPTRG(lw));
      lw = 0;
    }
    lwast = ADD_LWAST(dtype, i);
    if (lwast && A_TYPEG(lwast) == A_ID &&
        (CCSYMG(A_SPTRG(lwast)) || HCCSYMG(A_SPTRG(lwast)))) {
      /* store lower bound into the variable */
      if (lw != lwast) {
        if (lw == 0) {
          rilm = plower("oS", lowersym.bnd.con, lower_getiszcon(lwval));
        } else {
          lower_expression(lw);
          rilm = lower_ilm(lw);
          rilm = lower_conv_ilm(lw, rilm, A_NDTYPEG(lw), A_DTYPEG(lwast));
        }
        lower_expression(lwast);
        lilm = lower_base(lwast);
        lower_typestore(A_DTYPEG(lwast), lilm, rilm);
      }
    }

    if (subscripts == -1) {
      up = ADD_UPBD(dtype, i);
    } else {
      up = ASD_SUBS(subscripts, i);
      if (up && A_TYPEG(up) == A_TRIPLE) {
        up = A_UPBDG(up);
      }
    }
    if (up == 0) {
      up = -1;
    } else if (A_TYPEG(up) == A_CNST) {
      upval = ad_val_of(A_SPTRG(up));
      up = 0;
    }
    upast = ADD_UPAST(dtype, i);
    if (upast && A_TYPEG(upast) == A_ID &&
        (CCSYMG(A_SPTRG(upast)) || HCCSYMG(A_SPTRG(upast)))) {
      if (up != -1 && up != upast) {
        if (up == 0) {
          rilm = plower("oS", lowersym.bnd.con, lower_getiszcon(upval));
        } else {
          lower_expression(up);
          rilm = lower_ilm(up);
          rilm = lower_conv_ilm(up, rilm, A_NDTYPEG(up), A_DTYPEG(upast));
        }
        lower_expression(upast);
        lilm = lower_base(upast);
        lower_typestore(A_DTYPEG(upast), lilm, rilm);
      }
    }

    if (subscripts == -1) {
      stride = -1;
    } else {
      stride = ASD_SUBS(subscripts, i);
      if (stride && A_TYPEG(stride) == A_TRIPLE) {
        stride = A_STRIDEG(stride);
      } else {
        stride = -1;
      }
    }
    if (stride <= 0) {
      stride = -1;
    } else if (A_TYPEG(stride) == A_CNST) {
      strideval = ad_val_of(A_SPTRG(stride));
      stride = 0;
    }

    /* update array zero-base offset */
    if (zbaseilm == 0 && mlpyrilm == 0 && lw == 0 && up == 0) {
      /* everything is constant */
      if (stride == 0 && strideval > 0) {
        zbaseval = zbaseval + mlpyrval * lwval;
      } else if (stride == 0 && strideval < 0) {
        zbaseval = zbaseval + mlpyrval * upval;
      } else {
        zbaseval = zbaseval + mlpyrval * (lwval < upval ? lwval : upval);
      }
    } else if (mlpyrilm >= 0) {
      /* must insert ILMs here to update zbase */
      /* zbaseilm = old-zbaseilm + mlpyr * lower */
      if (mlpyrilm == 0 && lw == 0) {
        rilm =
            plower("oS", lowersym.bnd.con, lower_getiszcon(mlpyrval * lwval));
      } else {
        if (lw == 0) {
          rilm = plower("oS", lowersym.bnd.con, lower_getiszcon(lwval));
        } else {
          lower_expression(lw);
          rilm = lower_ilm(lw);
          rilm = lower_conv_ilm(lw, rilm, A_NDTYPEG(lw), lowersym.bnd.dtype);
        }
        if (mlpyrilm != 0) {
          rilm = plower("oii", lowersym.bnd.mul, mlpyrilm, rilm);
        } else if (mlpyrval != 1) {
          int ilm;
          ilm = plower("oS", lowersym.bnd.con, lower_getiszcon(mlpyrval));
          rilm = plower("oii", lowersym.bnd.mul, ilm, rilm);
        }
      }
      if (zbaseilm != 0) {
        zbaseilm = plower("oii", lowersym.bnd.add, zbaseilm, rilm);
      } else if (zbaseval == 0) {
        zbaseilm = rilm;
      } else {
        zbaseilm = plower("oS", lowersym.bnd.con, lower_getiszcon(zbaseval));
        zbaseilm = plower("oii", lowersym.bnd.add, zbaseilm, rilm);
      }
    }

    /* update multiplier for next dimension;
     * mlpyr = mlpyr * (upper - lower + 1) */
    if (lw == 0 && up == 0 && mlpyrilm == 0) {
      /* simple constants */
      if (stride == 0 && strideval > 0) {
        mlpyrval *= (upval >= lwval) ? upval - lwval + 1 : 0;
      } else if (stride == 0 && strideval < 0) {
        mlpyrval *= (upval <= lwval) ? lwval - upval + 1 : 0;
      } else {
        mlpyrval *=
            (upval >= lwval) ? (upval - lwval + 1) : (lwval - upval + 1);
      }
    } else if (up < 0 || mlpyrilm < 0) {
      /* unknown upper bound, can't compute */
      mlpyrilm = -1;
      mlpyrbaseilm = 0;
    } else {
      int xilm;
      if (up == 0 && lw == 0) {
        xilm =
            plower("oS", lowersym.bnd.con, lower_getiszcon(upval - lwval + 1));
      } else {
        int lwilm;
        if (up == 0) {
          xilm = plower("oS", lowersym.bnd.con, lower_getiszcon(upval));
        } else {
          lower_expression(up);
          xilm = lower_ilm(up);
          xilm = lower_conv_ilm(up, xilm, A_NDTYPEG(up), lowersym.bnd.dtype);
        }
        if (lw == 0) {
          if (lwval == 1) {
            lwilm = 0;
          } else {
            lwilm = plower("oS", lowersym.bnd.con, lower_getiszcon(lwval - 1));
          }
        } else {
          lower_expression(lw);
          lilm = lower_ilm(lw);
          lilm = lower_conv_ilm(up, lilm, A_NDTYPEG(lw), lowersym.bnd.dtype);
          rilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
          lwilm = plower("oii", lowersym.bnd.sub, lilm, rilm);
        }
        if (lwilm != 0) {
          xilm = plower("oii", lowersym.bnd.sub, xilm, lwilm);
        }
      }
      if (mlpyrilm > 0) {
        xilm = plower("oii", lowersym.bnd.mul, xilm, mlpyrilm);
      } else if (mlpyrval != 1) {
        int ilm;
        ilm = plower("oS", lowersym.bnd.con, lower_getiszcon(mlpyrval));
        xilm = plower("oii", lowersym.bnd.mul, xilm, ilm);
      }
      mlpyrilm = xilm;
      mlpyrbaseilm = 0;
    }
    /* store multiplier value in next multiplier variable */
    m = ADD_MLPYR(dtype, i + 1);
    if (m == 0) {
      /* nowhere to store */
    } else if (A_TYPEG(m) == A_CNST && subscripts == 0) {
      assert(mlpyrilm == 0 && mlpyrval == ad_val_of(A_SPTRG(m)),
             "put_adjarr_bounds: multiplier doesn't match", dtype, 3);
    } else if (mlpyrilm >= 0 && A_TYPEG(m) == A_ID &&
               (CCSYMG(A_SPTRG(m)) || HCCSYMG(A_SPTRG(m)))) {
      int xilm;
      lower_expression(m);
      xilm = lower_base(m);
      if (mlpyrilm > 0) {
        lower_typestore(A_DTYPEG(m), xilm, mlpyrilm);
        mlpyrbaseilm = xilm;
        mlpyrtype = A_DTYPEG(m);
        nelem = m;
      } else if (mlpyrilm == 0) {
        int yilm;
        yilm = plower("oS", lowersym.bnd.con, lower_getiszcon(mlpyrval));
        lower_typestore(A_DTYPEG(m), xilm, yilm);
      }
    }
  }
  zbase = ADD_ZBASE(dtype);
  if (zbase && A_TYPEG(zbase) == A_ID &&
      (CCSYMG(A_SPTRG(zbase)) || HCCSYMG(A_SPTRG(zbase)))) {
    int lilm;
    if (zbaseilm == 0) {
      zbaseilm = plower("oS", lowersym.bnd.con, lower_getiszcon(zbaseval));
    }
    lower_expression(zbase);
    lilm = lower_base(zbase);
    lower_typestore(A_DTYPEG(zbase), lilm, zbaseilm);
  }
  /* numelm is stored by the 'n' value of ADD_MLPYR */
  if (mlpyrilm != 0 && pcomputednumelm != NULL) {
    if (mlpyrbaseilm) {
      if (size_of(A_DTYPEG(nelem)) < 8) {
        nelem = mk_convert(nelem, DT_INT8);
        lower_expression(nelem);
        mlpyrbaseilm = lower_ilm(nelem);
      }
      *pcomputednumelm = mlpyrbaseilm;
    } else {
      if (size_of(lowersym.bnd.dtype) < 8) {
        mlpyrilm = plower("oi", "ITOI8", mlpyrilm);
      }
      *pcomputednumelm = mlpyrilm;
    }
  }
} /* put_adjarr_bounds */

/** \brief Look at the formal arguments; for array arguments, get the actual
           argument, call put_adjarr_bounds to fill in the .A variables
 */
void
fill_entry_bounds(int sptr, int lineno)
{
  int i, dpdsc, paramct;
  paramct = PARAMCTG(sptr);
  dpdsc = DPDSCG(sptr);
  lower_start_stmt(lineno, 0, FALSE, 0);
  for (i = 0; i < paramct; ++i) {
    int param;
    param = aux.dpdsc_base[dpdsc + i];
    if (param) {
      int dtype;
      if (NEWARGG(param)) {
        param = NEWARGG(param);
      }
      dtype = DTYPEG(param);
      if (DTY(dtype) == TY_ARRAY) {
        if (LNRZDG(param) && DTY(dtype - 1) < 0) {
          dtype = -DTY(dtype - 1);
        }
        put_adjarr_bounds(dtype, -1, NULL);
      }
    }
  }
  lower_end_stmt(0);
} /* fill_entry_bounds */

static void
fill_midnum(int sptr)
{
  int midnum;
  midnum = getccsym('Z', sptr, ST_VAR);
  SCP(midnum, SC_LOCAL);
  if (!XBIT(125, 0x2000)) {
    DTYPEP(midnum, DT_ADDR);
  } else {
    /* construct pointer datatype */
    int ndtype;
    ndtype = get_type(2, TY_PTR, DTYPEG(sptr));
    DTYPEP(midnum, ndtype);
  }
  MIDNUMP(sptr, midnum);
  SCP(sptr, SC_BASED);
} /* fill_midnum */

/* The parent should be a struct, derived type, or union,
 * which contains sptr as one of its members.
 * If not, some intermediate anonymous struct member names are needed. */
static int
intermediate_members(int base, int parent, int sptr)
{
  int p = LOWER_MEMBER_PARENT(sptr);
  if (p) {
    int a;
    switch (A_TYPEG(parent)) {
    case A_MEM:
      a = A_MEMG(parent);
      if (A_SPTRG(a) == p)
        return base;
      break;
    case A_ID:
      if (A_SPTRG(parent) == p)
        return base;
      break;
    default:
      break;
    }
    base = intermediate_members(base, parent, p);
    base = plower("oiS", "MEMBER", base, p);
  }
  return base;
} /* intermediate_members */

static int lower_numelm(int, int, int, int);
static int numelm_constant; /* set by lower_numelm() */

static int lower_sptr(int sptr, int);
static int lower_base_address(int ast, int pointerval);

int
lower_replacement(int ast, int sym)
{
  int base, lbase;
  if (!sym)
    return 0;
  lower_visit_symbol(sym);
  if (A_TYPEG(ast) == A_SUBSCR) {
    ast = A_LOPG(ast);
  }
  if (STYPEG(sym) != ST_MEMBER || A_TYPEG(ast) != A_MEM) {
    return lower_sptr(sym, VarBase);
  }
  lower_expression(A_PARENTG(ast));
  lbase = lower_ilm(A_PARENTG(ast));
  lbase = intermediate_members(lbase, A_PARENTG(ast), sym);
  base = plower("oiS", "MEMBER", lbase, sym);
  return base;
} /* lower_replacement */

static void
add_nullify(int ast)
{
  int sym, sdsc, lilm, rilm, silm, ptr, dtype;
  sym = find_pointer_variable(ast);
  dtype = DTYPEG(sym);
  sdsc = SDSCG(sym);
  ptr = MIDNUMG(sym);
  lower_expression(ast);
  lower_disable_ptr_chk = 1;
  if (DTY(dtype) == TY_PTR) {
    /* scalar pointer, a la member */
    lilm = lower_target(ast);
  } else if (ptr == 0) {
    lilm = lower_base(ast);
  } else {
    lilm = lower_replacement(ast, ptr);
  }
  lower_disable_ptr_chk = 0;
  rilm = lower_null();
  lower_typestore(DT_ADDR, lilm, rilm);

  if (sdsc && STYPEG(sdsc) != ST_PARAM &&
      /* don't set 'sdsc' if pointer is member and sdsc is not */
      (STYPEG(sym) != ST_MEMBER || STYPEG(sdsc) == ST_MEMBER)) {
    int dtype;
    lilm = lower_replacement(ast, sdsc);
    dtype = DTYPEG(sdsc);
    if (DTY(dtype) == TY_ARRAY) {
      /* get first element */
      silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
      lilm = plower("onidi", "ELEMENT", 1, lilm, dtype, silm);
      dtype = DTY(dtype + 1);
    }
    rilm = plower("oS", "ICON", lowersym.intzero);
    lower_typestore(dtype, lilm, rilm);
  }
} /* add_nullify */

static void
lower_finish_subprogram(int rutype)
{
  switch (rutype) {
  case RU_BDATA:
    break;
  case RU_SUBR:
  case RU_FUNC:
    break;
  case RU_PROG:
    break;
  }
} /* lower_finish_subprogram */

/*
 * Functions declared within a MODULE specification section do not have
 * their own or their arguments' array datatype fields replaced by temps.
 * If they are passed as arguments to other subprograms, the datatype
 * must be also passed, and thus the datatype fields must be replaced.
 *
 * Some ubound/lbound asts are accesses to the runtime descrptor, do
 * not replace these.
 */
static void
fix_array_fields(int dtype)
{
  int numdim, i;
  if (DTY(dtype) != TY_ARRAY)
    return;
  numdim = ADD_NUMDIM(dtype);
  for (i = 0; i < numdim; ++i) {
    int lwast, upast;
    lwast = ADD_LWAST(dtype, i);
    if (lwast != 0) {
      if (A_ALIASG(lwast))
        lwast = A_ALIASG(lwast);
      if (A_TYPEG(lwast) != A_CNST && A_TYPEG(lwast) != A_ID &&
          !(A_TYPEG(lwast) == A_SUBSCR &&
            DESCARRAYG(find_array(lwast, NULL)))) {
        ADD_LWAST(dtype, i) = mk_shared_bnd_ast(lwast);
      }
    }
    upast = ADD_UPAST(dtype, i);
    if (upast != 0) {
      if (A_ALIASG(upast))
        upast = A_ALIASG(upast);
      if (A_TYPEG(upast) == A_BINOP && A_OPTYPEG(upast) == OP_ADD) {
        /* handle special case of lower+(extent-1) */
        int l, r, rl, rr;
        l = A_LOPG(upast);
        r = A_ROPG(upast);
        if (A_TYPEG(r) == A_BINOP && A_OPTYPEG(r) == OP_SUB) {
          rl = A_LOPG(r);
          rr = A_ROPG(r);
          if (A_TYPEG(l) == A_SUBSCR && A_TYPEG(rl) == A_SUBSCR &&
              A_TYPEG(rr) == A_CNST) {
            l = A_LOPG(l);
            rl = A_LOPG(rl);
            if (A_TYPEG(l) == A_ID && DESCARRAYG(A_SPTRG(l)) &&
                A_TYPEG(rl) == A_ID && DESCARRAYG(A_SPTRG(rl))) {
              upast = 0;
            }
          }
        }
      }
      if (upast && A_TYPEG(upast) != A_CNST && A_TYPEG(upast) != A_ID &&
          !(A_TYPEG(upast) == A_SUBSCR &&
            DESCARRAYG(find_array(upast, NULL)))) {
        ADD_UPAST(dtype, i) = mk_shared_bnd_ast(upast);
        if (ADD_EXTNTAST(dtype, i) == upast)
          ADD_EXTNTAST(dtype, i) = ADD_UPAST(dtype, i);
      }
    }
  }
} /* fix_array_fields */


/* if there are alternate return labels, convert to a function call */
static void
handle_arguments(int ast, int symfunc, int via_ptr)
{
  int count, args, paramcount, paramc, altreturn, i, ilm, ilm2, params;
  int dtype, dtproc, iface = 0;
  int callee;
  int psptr, prevsptr, sptr;
  int via_tbp, tbp_mem, tbp_pass_arg, tbp_bind;
  int tbp_nopass_arg, tbp_nopass_sdsc;
  int unlpoly; /* CLASS(*) */

  bool procDummyNeedsDesc = proc_arg_needs_proc_desc(symfunc);
  switch (A_TYPEG(A_LOPG(ast))) {
  case A_ID:
  case A_LABEL:
  case A_ENTRY:
  case A_SUBSCR:
  case A_SUBSTR:
  case A_MEM:
    tbp_mem = memsym_of_ast(A_LOPG(ast));
    if (CLASSG(tbp_mem) && CCSYMG(tbp_mem) && STYPEG(tbp_mem)) {
      tbp_pass_arg = sym_of_ast(A_LOPG(ast));
      tbp_bind = BINDG(tbp_mem);
      if (!INVOBJG(tbp_bind) && !NOPASSG(tbp_mem)) {
        get_tbp_argno(tbp_bind, TBPLNKG(tbp_mem));
      }
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    tbp_mem = tbp_pass_arg = tbp_bind = 0;
  }
#if DEBUG
  assert(!tbp_bind || STYPEG(tbp_bind) == ST_PROC,
         "handle_arguments: invalid stype for type bound procedure",
         STYPEG(tbp_bind), 4);
#endif

  if (CLASSG(tbp_bind) && VTOFFG(tbp_bind) &&
      (INVOBJG(tbp_bind) || NOPASSG(tbp_mem))) { /* NOPASS needs fixing */
    via_tbp = 1;
    if (NOPASSG(tbp_mem)) {
      tbp_nopass_arg = pass_sym_of_ast(A_LOPG(ast));
      tbp_nopass_sdsc =
          A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast)) : 0;
      if (!tbp_nopass_sdsc)
        tbp_nopass_sdsc = get_type_descr_arg(gbl.currsub, tbp_nopass_arg);
      lower_expression(A_LOPG(ast));
      tbp_nopass_arg = lower_base(A_LOPG(ast));
    } else {
      tbp_nopass_arg = tbp_nopass_sdsc = 0;
    }
  } else {
    via_tbp = 0;
  }

  count = A_ARGCNTG(ast);
  NEED(count, lower_argument, int, lower_argument_size, count + 10);
  args = A_ARGSG(ast);

  dtproc = 0;       /* shut purify up */
  callee = symfunc; /* shut purify up */
  if (!via_ptr) {
    paramcount = PARAMCTG(symfunc);
    params = DPDSCG(symfunc);
  } else {
    dtype = DTYPEG(symfunc);
#if DEBUG
    assert(DTY(dtype) == TY_PTR, "handle_arguments, expected TY_PTR dtype",
           symfunc, 4);
#endif
    dtproc = DTY(dtype + 1);
#if DEBUG
    assert(DTY(dtproc) == TY_PROC, "handle_arguments, expected TY_PROC dtype",
           symfunc, 4);
#endif
    lower_expression(A_LOPG(ast));

    callee = lower_base(A_LOPG(ast));

    iface = DTY(dtproc + 2);
    paramcount = DTY(dtproc + 3);
    params = DTY(dtproc + 4);
  }
  if (procDummyNeedsDesc) {
    lower_expression(A_LOPG(ast));

    callee = lower_base(A_LOPG(ast));
  }
  altreturn = 0;

  for (i = 0; i < count; ++i) {
    int a;
    a = ARGT_ARG(args, i);
    if (a == 0)
      continue;
    if (A_TYPEG(a) != A_LABEL)
      lower_expression(a);
    switch (A_TYPEG(a)) {
    case A_ID:
    case A_MEM:
    case A_SUBSCR:
    case A_CNST:
    case A_LABEL:
      break;
    default:
      lower_ilm(a);
    }
  }
  paramc = 0;
  sptr = 0;
  for (i = 0; i < count; ++i) {
    int a, param, byval;
    prevsptr = sptr;
    sptr = 0;
    a = ARGT_ARG(args, i);
    lower_argument[i] = 0;
    param = 0;
    if (paramc < paramcount) {
      param = aux.dpdsc_base[params + paramc];
      ++paramc;
      if (symfunc == gbl.currsub) {
        /* argument list was rewritten; use original argument */
        int nparam = NEWARGG(param);
        if (nparam)
          param = nparam;
      }
    }
    if (a == 0)
      continue;
    byval = 0;
    if (!byval)
      byval = get_byval(symfunc, param);
    if (byval) {
      switch (A_TYPEG(a)) {
        int dt;

      case A_LABEL:
        ++altreturn;
        break;
      case A_ID:
        /* for nonscalar identifiers, just pass by reference */
        sptr = A_SPTRG(a);

        switch (STYPEG(sptr)) {
        case ST_VAR:
        case ST_IDENT:
          if (param && POINTERG(param) && POINTERG(sptr))
            goto by_reference;
          break;
        default:
          goto by_reference;
        }
        goto by_value;
      case A_MEM:
        /* if the formal is a pointer, pass the pointer address,
         * otherwise pass the data base address */
        sptr = A_SPTRG(A_MEMG(a));
        if (param && POINTERG(param) && POINTERG(sptr))
          goto by_reference;
        FLANG_FALLTHROUGH;
      case A_INTR:
        if (is_iso_cloc(a)) {
          /* byval C_LOC(x) == regular pass by reference (X),
             no type checking
           */
          a = ARGT_ARG(A_ARGSG(a), 0);
          goto by_reference;
        }
        FLANG_FALLTHROUGH;
      default:
      /* expressions & scalar variables -- always emit BYVAL.
       * expand will take do the right thing for nonscalar
       * expressions.
       */
      by_value:
        dt = A_DTYPEG(a);
        if (A_OPTYPEG(a) == OP_REF) {
          ilm = lower_ilm(A_LOPG(a));
          lower_argument[i] = plower("oi", "DPREF", ilm);
          break;
        } else {
          ilm = lower_ilm(a);
        }
        if (DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR) {
          if (DTY(dt) == TY_CHAR)
            ilm = plower("oi", "ICHAR", ilm);
          else
            ilm = plower("oi", "INCHAR", ilm);
          if (DTY(stb.user.dt_int) == TY_INT8)
            ilm = plower("oi", "ITOI8", ilm);
          dt = stb.user.dt_int;
        }
        lower_argument[i] = plower("oid", "BYVAL", ilm, dt);
        break;
      }
      continue;
    }
  by_reference:
    unlpoly = 0;
    if (param && is_unl_poly(param)) {
      unlpoly = 1;
    }
    switch (A_TYPEG(a)) {
    case A_ID:
      /* if the formal is a pointer, pass the pointer address,
       * otherwise pass the data base address */
      sptr = A_SPTRG(a);
      if (param &&
          (
#ifdef RVALLOCG
              (RVALLOCG(MIDNUMG(sptr)) && ALLOCATTRG(FVALG(symfunc))) ||
#endif
              (POINTERG(param) && POINTERG(sptr)) ||
              (ALLOCATTRG(param) && ALLOCATTRG(sptr)))) {
        lower_disable_ptr_chk = 1;
        if (DTY(DTYPEG(sptr)) == TY_ARRAY && !XBIT(57, 0x80000)) {
          lower_argument[i] = lower_base(a);
        } else {
          ilm = lower_target(a);
          ilm2 = plower("oS", "BASE", sptr);
          lower_argument[i] = plower("oii", "PARG", ilm, ilm2);
        }
        lower_disable_ptr_chk = 0;
        }else if(POINTERG(sptr) && A_TYPEG(a) == A_ID && A_PTRREFG(a)) {

        /* Special handling of pointer to pointer arguments in runtime
         * routines.
         */
        
        lower_expression(a);
        lower_disable_ptr_chk = 1;
        if (DTY(DTYPEG(sptr)) == TY_PTR) {
          ilm = lower_target(a);
        } else {
          if (MIDNUMG(sptr) == 0) {
            ilm = lower_base(a);
          } else {
            ilm = lower_replacement(a, MIDNUMG(sptr));
          }
        }
        lower_argument[i] = ilm;
      } else {
        lower_argument[i] = lower_base(a);
      }
      switch (STYPEG(sptr)) {
      case ST_PROC:
      case ST_ENTRY:
      case ST_MODPROC:
        break;
      default:
        if (DTYPEG(sptr)) {
          lower_argument[i] =
              plower_arg("oid", lower_argument[i], DTYPEG(sptr), unlpoly);
        }
      }
      break;
    case A_MEM:
      /* if the formal is a pointer, pass the pointer address,
       * otherwise pass the data base address */
      sptr = A_SPTRG(A_MEMG(a));
      if (param && ((POINTERG(param) && POINTERG(sptr)) ||
                    (ALLOCATTRG(param) && ALLOCATTRG(sptr)))) {
        lower_disable_ptr_chk = 1;
        if (DTY(DTYPEG(sptr)) == TY_ARRAY && !XBIT(57, 0x80000)) {
          lower_argument[i] = lower_base(a);
        } else {
          ilm = lower_target(a);
          ilm2 = plower("oS", "BASE", sptr);
          lower_argument[i] = plower("oii", "PARG", ilm, ilm2);
        }
        lower_disable_ptr_chk = 0;
      } else {
        lower_argument[i] = lower_base(a);
      }
      lower_argument[i] =
          plower_arg("oid", lower_argument[i], DTYPEG(sptr), unlpoly);
      break;
    case A_SUBSCR:
    case A_CNST:
      lower_argument[i] = lower_base(a);
      if (A_DTYPEG(a)) {
        lower_argument[i] =
            plower_arg("oid", lower_argument[i], A_DTYPEG(a), unlpoly);
      }
      break;
    case A_LABEL:
      ++altreturn;
      break;
    case A_FUNC:
      lower_argument[i] = lower_parenthesize_expression(a);
      if (A_DTYPEG(a)) {
        if (param && OPTARGG(param) && a == astb.ptr0) {
          /* missing optional argument */
          /* treat like the datatype of the optional dummy */
          lower_argument[i] =
              plower_arg("oiD", lower_argument[i], DTYPEG(param), unlpoly);
        } else {
          int farg_dt;
          int sym;
          farg_dt = A_DTYPEG(a);
          switch (A_TYPEG(A_LOPG(a))) {
          case A_ID:
          case A_LABEL:
          case A_ENTRY:
          case A_SUBSCR:
          case A_SUBSTR:
          case A_MEM:
            sym = memsym_of_ast(A_LOPG(a));
            if (CLASSG(sym) && VTABLEG(sym) && BINDG(sym)) {
              sym = VTABLEG(sym);
              break;
            }
            FLANG_FALLTHROUGH;
          default:
            sym = 0;
          }
          if (sym) {
            if (ELEMENTALG(sym)) {
              if (DTY(farg_dt) == TY_ARRAY) {
                /* should be array, this could be an assert */
                farg_dt = DTY(farg_dt + 1);
              }
            }
          } else if (ELEMENTALG(A_SPTRG(A_LOPG(a)))) {
            if (DTY(farg_dt) == TY_ARRAY) {
              /* should be array, this could be an assert */
              farg_dt = DTY(farg_dt + 1);
            }
          }
          lower_argument[i] =
              plower_arg("oid", lower_argument[i], farg_dt, unlpoly);
        }
      }
      break;
    default:
      lower_argument[i] = lower_parenthesize_expression(a);
      if (A_DTYPEG(a)) {
        if (param && OPTARGG(param) && a == astb.ptr0) {
          /* missing optional argument */
          /* treat like the datatype of the optional dummy */
          lower_argument[i] = plower_arg("oiD", lower_argument[i], 0, 0);
        } else {
          lower_argument[i] =
              plower_arg("oid", lower_argument[i], A_DTYPEG(a), unlpoly);
        }
      }
      break;
    }
  }
  if (via_tbp) {
    int a;
    int sdsc;
    paramcount = PARAMCTG(symfunc);
    if (altreturn) {
      ilm = plower("om", "IUVFUNCA");
      plower("nnm", count - altreturn,
             (CFUNCG(symfunc) || (iface && CFUNCG(iface))) ? 1 : 0);

      i = find_dummy_position(symfunc, PASSG(tbp_mem));
      if (i <= 0)
        i = max_binding_invobj(symfunc, INVOBJG(tbp_bind)) - 1;

      VTABLEP(tbp_mem, symfunc);
      plower("sm", tbp_mem);
      if (!NOPASSG(tbp_mem) && i >= 0) {
        plower("im", lower_argument[i]);
        a = ARGT_ARG(args, i);
        sdsc = A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast)) : 0;
        if (!sdsc)
          sdsc = get_type_descr_arg(gbl.currsub, memsym_of_ast(a));
        plower("sm", sdsc);
      } else {

        assert(NOPASSG(tbp_mem),
               "handle_arguments: nopass not set for type bound procedure",
               STYPEG(tbp_bind), 4);

        plower("im", tbp_nopass_arg);
        plower("sm", tbp_nopass_sdsc);
      }
    } else {
      int currsub;

      plower("om", "UVCALLA");
      plower("nnm", count,
             (CFUNCG(symfunc) || (iface && CFUNCG(iface))) ? 1 : 0);

      i = find_dummy_position(symfunc, PASSG(tbp_mem));
      if (i <= 0)
        i = INVOBJG(tbp_bind);
      --i;
      a = ARGT_ARG(args, i);

      if (A_TYPEG(a) == A_ID && !NOPASSG(tbp_mem) && FVALG(symfunc) &&
          (POINTERG(FVALG(symfunc)) || ALLOCATTRG(FVALG(symfunc))) &&
          !RESULTG(aux.dpdsc_base[params])) {
        /* Check to see if we have not inserted the result argument yet.
         * This can happen when we call a tbp's module procedure that returns
         * an allocatable or pointer from another module procedure located
         * in the same module. In other words, the INVOBJ field for the
         * tbp has not been adjusted yet. This occurs later in semfin()
         * when it calls incr_invobj_for_retval_add() function. For now,
         * just increment i.
         */
        ++i;
      }

      VTABLEP(tbp_mem, symfunc);
      plower("sm", tbp_mem);
      if (!NOPASSG(tbp_mem) && i >= 0) {
        plower("im", lower_argument[i]);
        a = ARGT_ARG(args, i);
        currsub = gbl.currsub;
        psptr = sym_of_ast(a);
        if (SCG(psptr) == SC_DUMMY && !is_arg_in_entry(currsub, psptr)) {
          /* gbl.currsub (better be) a contained subprogram. */
          if (SCOPEG(currsub) && STYPEG(SCOPEG(currsub)) == ST_ALIAS &&
              STYPEG(SYMLKG(SCOPEG(currsub))) != ST_MODULE) {
            currsub = SYMLKG(SCOPEG(currsub));
          }
        }
        sdsc = A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast)) : 0;
        if (!sdsc)
          sdsc = get_type_descr_arg(currsub, memsym_of_ast(a));
        plower("sm", sdsc);
      } else {

        assert(NOPASSG(tbp_mem),
               "handle_arguments: nopass not set for type bound procedure",
               STYPEG(tbp_bind), 4);

        plower("im", tbp_nopass_arg);
        plower("sm", tbp_nopass_sdsc);
      }
    }
  } else if (!via_ptr && !procDummyNeedsDesc) {
    if (altreturn) {
      ilm = plower("onm", "IUFUNC", count - altreturn);
    } else {
      plower("onm", "UCALL", count);
    }
    plower("sm", symfunc);
    paramcount = PARAMCTG(symfunc);
  } else {
    if (altreturn) {
      if (is_procedure_ptr(symfunc) || procDummyNeedsDesc) {
        int sdsc = A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast))
                                         : SDSCG(symfunc);
        ilm = plower("om", "PIUFUNCA");
        plower("nnsim", count - altreturn,
               (CFUNCG(symfunc) || (iface && CFUNCG(iface))) ? 1 : 0, sdsc,
               callee);
      } else {
        ilm = plower("om", "IUFUNCA");
        plower("nnim", count - altreturn,
               (CFUNCG(symfunc) || (iface && CFUNCG(iface))) ? 1 : 0, callee);
      }
    } else {
      if (SDSCG(symfunc) == 0) {
        plower("om", "UCALLA");
        plower("nnim", count,
               (CFUNCG(symfunc) || (iface && CFUNCG(iface))) ? 1 : 0, callee);
      } else {
        int sdsc = A_INVOKING_DESCG(ast) ? sym_of_ast(A_INVOKING_DESCG(ast))
                                         : SDSCG(symfunc);
        plower("om", "UPCALLA");
        plower("nnsim", count,
               (CFUNCG(symfunc) || (iface && CFUNCG(iface))) ? 1 : 0, sdsc,
               callee);
      }
    }
    paramcount = DTY(dtproc + 3);
  }
  for (i = 0; i < count; ++i) {
    int a, param;
    a = ARGT_ARG(args, i);
    param = 0;
    if (params && i < paramcount) {
      param = aux.dpdsc_base[params + i];
    }
    if (a && A_TYPEG(a) != A_LABEL) {
      fix_array_fields(A_NDTYPEG(a));
      if (a == astb.ptr0 && param && OPTARGG(param)) {
        plower("Am", lower_argument[i], DTYPEG(param));
      } else {
        plower("am", lower_argument[i], A_NDTYPEG(a));
      }
    }
  }
  plower("C", symfunc);
  if (altreturn) {
    int lab, labnum;
    /* put out list of labels for alternate return */
    lab = lower_lab();
    labnum = plabel(lab, altreturn, 1);
    altreturn = 0;
    for (i = 0; i < count; ++i) {
      int a;
      a = ARGT_ARG(args, i);
      if (A_TYPEG(a) == A_LABEL) {
        ++altreturn;
        plabel(A_SPTRG(a), altreturn, 0);
      }
    }
    /* and the computed goto that does the work */
    plower("oin", "CGOTO", ilm, labnum);
    plower("oL", "LABEL", lab);
    DTYPEP(symfunc, DT_INT4);
  }
} /* handle_arguments */

static int
compute_dotrip(int std, int initincsame, int doinitilm, int doendilm, int doinc,
               int doincilm, int dtype, int dotrip)
{
  int ilm, dotripilm;
  ilm = doendilm;
  if (initincsame) {
    /* trip count is (UB-LB+inc)/inc, but LB == inc */
  } else {
    /* trip count is (UB-LB+inc)/inc */
    ilm = plower("oii", ltyped("SUB", dtype), ilm, doinitilm);
    if (doinc && STYPEG(doinc) == ST_CONST) {
      doincilm = plower("oS", ltyped("CON", dtype), doinc);
    }
    ilm = plower("oii", ltyped("ADD", dtype), ilm, doincilm);
  }
  if (doinc != stb.i1 && doinc != stb.flt1 && doinc != stb.dbl1) {
    if (doinc && STYPEG(doinc) == ST_CONST) {
      doincilm = plower("oS", ltyped("CON", dtype), doinc);
    }
    ilm = plower("oii", ltyped("DIV", dtype), ilm, doincilm);
  }
  if (XBIT(68, 0x1)) {
    if (dtype == DT_INT8)
      dotripilm = lower_conv_ilm(0, ilm, dtype, DT_INT8);
    else
      dotripilm = lower_conv_ilm(0, ilm, dtype, DT_INT4);
  } else {
    if (XBIT(49, 0x100) && dtype == DT_INT8)
      dotripilm = lower_conv_ilm(0, ilm, dtype, DT_INT8);
    else
      dotripilm = lower_conv_ilm(0, ilm, dtype, DT_INT4);
  }
  if (dotrip) {
    ilm = plower("oS", "BASE", dotrip);
    lower_typestore(DTYPEG(dotrip), ilm, dotripilm);
  }
  return dotripilm;
} /* compute_dotrip */

#ifdef FLANG_LOWERILM_UNUSED
/* Hacked compute_dotrip() where the dotripilm is not converted to DT_INT4 */
static int
compute_dotrip8(int std, int initincsame, int doinitilm, int doendilm,
                int doinc, int doincilm, int dtype, int dotrip)
{
  int ilm, dotripilm;
  ilm = doendilm;
  if (initincsame) {
    /* trip count is (UB-LB+inc)/inc, but LB == inc */
  } else {
    /* trip count is (UB-LB+inc)/inc */
    ilm = plower("oii", ltyped("SUB", dtype), ilm, doinitilm);
    if (doinc && STYPEG(doinc) == ST_CONST) {
      doincilm = plower("oS", ltyped("CON", dtype), doinc);
    }
    ilm = plower("oii", ltyped("ADD", dtype), ilm, doincilm);
  }
  if (doinc != stb.i1 && doinc != stb.flt1 && doinc != stb.dbl1) {
    if (doinc && STYPEG(doinc) == ST_CONST) {
      doincilm = plower("oS", ltyped("CON", dtype), doinc);
    }
    ilm = plower("oii", ltyped("DIV", dtype), ilm, doincilm);
  }
  dotripilm = ilm;
  if (dotrip) {
    ilm = plower("oS", "BASE", dotrip);
    lower_typestore(DTYPEG(dotrip), ilm, dotripilm);
  }
  return dotripilm;
} /* compute_dotrip8 */
#endif

static int
dotemp(char letter, int dtype, int std)
{
  int temp, stype;
  char pf[3];
  if (DTY(dtype) == TY_ARRAY) {
    stype = ST_ARRAY;
  } else {
    stype = ST_VAR;
  }
  pf[0] = 'd';
  pf[1] = letter;
  pf[2] = '\0';
  if (STD_PAR(std) || STD_TASK(std)) {
    temp = getccssym_sc(pf, lowersym.docount, stype, SC_PRIVATE);
    if (STD_TASK(std))
      TASKP(temp, 1);
  } else {
    temp = getccssym_sc(pf, lowersym.docount, stype, SC_LOCAL);
  }
  DTYPEP(temp, dtype);
  return temp;
} /* dotemp */

static void
set_mp_loop_var(int var, int rilm, int dtype)
{
  int dest, lilm;

  dest = mk_id(var);
  lower_expression(dest);
  lilm = lower_base(dest);
  lilm = lower_conv_ilm(dest, lilm, A_NDTYPEG(dest), dtype);

  if (DTYG(dtype) == DT_INT8)
    plower("oii", "KST", lilm, rilm);
  else
    plower("oii", "IST", lilm, rilm);
}

static void
set_loop_bound(int newupper, int oilm, int nilm, int dtype, int labo,
               int increment)
{
  int ilm;
  /* if (increment)
   *     newupper = min(oldupper, newupper)
   *   goto labn
   * else
   *     newupper = max(oldupper, newupper)
   * labn:
   *
   */

  if (increment == 1) {
    nilm = plower("oii", ltyped("MAX", dtype), nilm, oilm);
    ilm = lower_sptr(newupper, VarBase);
    ilm = lower_typestore(DTYPEG(newupper), ilm, nilm);
    plower("oS", "BR", labo);

  } else {
    nilm = plower("oii", ltyped("MIN", dtype), nilm, oilm);
    ilm = lower_sptr(newupper, VarBase);
    ilm = lower_typestore(DTYPEG(newupper), ilm, nilm);
  }
}

static void
check_loop_bound(int lower, int upper, int oupper, int stride, int label,
                 int setbound, int incr_loop)
{
  int ilm, lilm, uilm, silm, bilm, oilm;
  int labo, labn;
  /* incr_loop == 0, don't know if loop incr or decr
                     need to check at runtime if stride is > 0
                     to determine if that is incror decr loop
   * incr_loop == 1, loop increment
   * incr_loop == -1 loop decrment
   *
   * if (stride > 0)      -- increment
   *   if (lower < oupper) -- decrement
   *     goto label
   *   goto labn
   * labo:
   *   if (lower > oupper)
   *     goto label
   * labn:
   *
   */

  lilm = plower("oS", "BASE", lower);
  lilm = lower_typeload(DTYPEG(lower), lilm);

  uilm = plower("oS", "BASE", upper);
  uilm = lower_typeload(DTYPEG(upper), uilm);

  silm = plower("oS", "BASE", stride);
  silm = lower_typeload(DTYPEG(stride), silm);

  oilm = plower("oS", "BASE", oupper);
  oilm = lower_typeload(DTYPEG(oupper), oilm);

  labo = lower_lab();
  labn = lower_lab();

  if (incr_loop == 0) {
    /* compare stride with  0 at runtime */
    if (DTYPEG(stride) != DT_INT8) {
      ilm = plower("oS", "ICON", lowersym.intzero);
      ilm = plower("oii", "ICMP", silm, ilm);
    } else {
      ilm = plower("oS", "KCON", lowersym.intzero);
      ilm = plower("oii", "KCMP", silm, ilm);
    }
  }

  /* check bound */
  if (DTYPEG(lower) != DT_INT8) {
    bilm = plower("oii", "ICMP", lilm, oilm);
  } else {
    bilm = plower("oii", "KCMP", lilm, oilm);
  }

  if (incr_loop == 0) {
    /* if (stride < 0) goto labo */
    ilm = plower("oi", "LT", ilm);
    plower("oiS", "BRT", ilm, labo);
  }

  /* do - if stride > 0 */
  if (incr_loop >= 0) {
    ilm = plower("oi", "GT", bilm);
    plower("oiS", "BRT", ilm, label);
    if (setbound)
      set_loop_bound(upper, oilm, uilm, DTYPEG(oupper), labn, 0);
  }

  if (incr_loop == 0)
    plower("oS", "BR", labn);

  /* do -- if stride < 0 */
  if (incr_loop == 0)
    plower("oL", "LABEL", labo);

  if (incr_loop <= 0) {
    ilm = plower("oi", "LT", bilm);
    plower("oiS", "BRT", ilm, label);
    if (setbound)
      set_loop_bound(upper, oilm, uilm, DTYPEG(oupper), labn, 1);
  }

  plower("oL", "LABEL", labn);
}

static void
llvm_omp_sched(int std, int ast, int dtype, int dotop, int dobottom, int dovar,
               int plast, int dotrip, int doinitilm, int doinc, int doincilm,
               int doendilm, int schedtype, int lineno)
{
  int chunkilm, odovar, newdovar;
  int itop, ibottom, chunkast;
  int newend, dost, ilm, o_ub, o_lb, ub, dotripilm, dyn;
  int is_dist = 0;
  int incr_loop = 0; /* 0: don't know, -1: decrement,  1: increment */
  int chunkone = 0;
  int tmpsched;

  if (doinc && STYPEG(doinc) == ST_CONST) {
    incr_loop = -1;
    if (doinc == stb.i1 || doinc == stb.k1) {
      incr_loop = 1;
    } else if (DTYPEG(doinc) == DT_INT) {
      if (_i4_cmp(doinc, stb.i0) > 0)
        incr_loop = 1;
    } else if (DTYPEG(doinc) == DT_INT8) {
      if (_i8_cmp(doinc, stb.k0) > 0)
        incr_loop = 1;
    }
  } else {
    /* fixme: check doincilm but what do we have cases where doinc is 0? */
  }

  /* pass to back end
   * (0x001 | chunked | ordered )for dynamic/guided/runtime/auto
   * (0x000 | chunked | blk_cyc)  for block cyclic
   * (0x000 | chunked | chunk_1)  for cyclic (chunk=1)
   * (0x000 | ordered)  for all static ordered
   *  0x000 for static(default)
   *
   * all ordered loops are 0x400 | ordered | chunked
   */

  if (A_DISTRIBUTEG(ast)) {
    is_dist = 1; /* distribute loop of distribute parallel do construct */
  } else if (A_DISTPARDOG(ast)) {
    is_dist = 2; /* parallel do of distribute parallel do construct */
  }

  dyn = 0;
  tmpsched = (schedtype & MP_SCH_TYPE_MASK);
  if ((tmpsched == MP_SCH_DYNAMIC) || (tmpsched == MP_SCH_GUIDED) ||
      (tmpsched == MP_SCH_RUNTIME) || (tmpsched == MP_SCH_AUTO) ||
      (schedtype & MP_SCH_ATTR_ORDERED)) {
    dyn = 1;
  }

  /* set loop index value */
  set_mp_loop_var(dovar, doinitilm, dtype);

  /* original upper bound - unchanged */
  o_ub = dotemp('u', dtype, std);
  set_mp_loop_var(o_ub, doendilm, dtype);

  /* loop upper bound - may change when call kmpc runtime */
  newend = dotemp('e', dtype, std);
  set_mp_loop_var(newend, doendilm, dtype);

  if (doinc == 0 || STYPEG(doinc) != ST_VAR || SCG(doinc) != SC_PRIVATE) {
    /* convert and store in a temp */
    int tdoinc = dotemp('i', dtype, std);
    if (doinc == 0) {
      assert(doincilm, "lower_do_stmt: doincilm", doincilm, 3);
      ilm = lower_sptr(tdoinc, VarBase);
      lower_typestore(dtype, ilm, doincilm);
    } else {
      doincilm = plower("oS", ltyped("CON", dtype), doinc);
      set_mp_loop_var(tdoinc, doincilm, dtype);
    }
    doinc = tdoinc;
  }
  /* keep original loop stride - stride can be changed by kmpc */
  ilm = plower("oS", "BASE", doinc);
  ilm = lower_typeload(dtype, ilm);
  dost = dotemp('s', dtype, std);
  set_mp_loop_var(dost, ilm, dtype);

  /* loop chunk */
  chunkast = A_CHUNKG(ast);
  if (chunkast == 0) {
    if (dtype == DT_INT8)
      chunkast = astb.k1;
    else
      chunkast = astb.i1;
  } else if (A_TYPEG(chunkast) == A_CNST) {
    int chunksptr = A_SPTRG(chunkast);
    if (chunksptr == stb.i1 || chunksptr == stb.k1)
      chunkone = 1;
  }
  lower_expression(chunkast);
  chunkilm = lower_conv(chunkast, dtype);

  odovar = dotemp('X', dtype, std);
  {
    if (dyn == 1) {
      newdovar = dotemp('x', dtype, std);
      set_mp_loop_var(newdovar, doinitilm, dtype);
      o_lb = dotemp('l', dtype, std);
      set_mp_loop_var(o_lb, doinitilm, dtype);
      ub = dotemp('U', dtype, std);
      set_mp_loop_var(ub, doendilm, dtype);
    } else {
      o_lb = dotemp('l', dtype, std);
      set_mp_loop_var(o_lb, doinitilm, dtype);
      ub = newend;
      newdovar = odovar;
    }
  }

  plower("osssssdn", "MPLOOP", o_lb, ub, dost, A_SPTRG(chunkast), plast, dtype,
         schedtype);

  /* dotop: */
  /* kmpc_dispatch_next will change dovar to 0 when we call after finish the
   * last iteration
   * so we need to pass temp lower bound so that we keep the iteration value the
   * same
   * we need this if loop index is lastprivate.
   */
  if (dyn == 1) {
    int lilm;
    plower("oL", "LABEL", dotop);

    ilm = plower("ossssd", "MPSCHED", newdovar, newend, dost, plast, dtype);
    lilm = plower("oS", "ICON", lowersym.intzero);
    ilm = plower("oii", "ICMP", ilm, lilm);
    ilm = plower("oi", "EQ", ilm);

    plower("oiS", "BRT", ilm, dobottom);
    {
      int lilm;
      doinitilm = plower("oS", "BASE", newdovar);
      doinitilm = lower_typeload(dtype, doinitilm);
      lilm = lower_sptr(odovar, VarBase);
      lower_typestore(dtype, lilm, doinitilm);
    }

    /* dovar = odovar */
    doinitilm = plower("oS", "BASE", odovar);
    doinitilm = lower_typeload(dtype, doinitilm);
    ilm = lower_sptr(dovar, VarBase);
    lower_typestore(dtype, ilm, doinitilm);

    /* lpct = ((lb-ub)+doinc)/doinc */
    doendilm = plower("oS", "BASE", newend);
    doendilm = lower_typeload(dtype, doendilm);

    doincilm = plower("oS", "BASE", dost);
    doincilm = lower_typeload(dtype, doincilm);

    dotripilm = compute_dotrip(std, 0, doinitilm, doendilm, dost, doincilm,
                               dtype, dotrip);

    /* DOBEG(dotrip, lab, lab) */
    itop = lower_lab();
    ibottom = lower_lab();
    if (((schedtype & MP_SCH_ATTR_CHUNKED) & MP_SCH_ATTR_ORDERED) ==
        MP_SCH_ATTR_CHUNKED) {
      /* do nothing */
    } else {
      plower_pdo(itop, schedtype);
      ilm = lower_typeload(DT_INT4, ilm);
      plower("oisS", "DOBEG", ilm, ibottom, dotrip);
    }
    plower("oL", "LABEL", itop);
    lower_end_stmt(std);

  } else {
    if (o_lb != dovar) {
      doinitilm = plower("oS", "BASE", o_lb);
      doinitilm = lower_typeload(dtype, doinitilm);
      ilm = lower_sptr(dovar, VarBase);
      lower_typestore(dtype, ilm, doinitilm);
    }

    /* odovar = dovar */
    doinitilm = plower("oS", "BASE", dovar);
    doinitilm = lower_typeload(dtype, doinitilm);
    ilm = lower_sptr(odovar, VarBase);
    lower_typestore(dtype, ilm, doinitilm);

    if (schedtype != 0) { 
      plower("oL", "LABEL", dotop);
      if (chunkone) {
        check_loop_bound(odovar, o_ub, o_ub, dost, dobottom, 0, incr_loop);
      } else {
        check_loop_bound(odovar, newend, o_ub, dost, dobottom, 1, incr_loop);
      }
    }

    /* dovar = odovar */
    doinitilm = plower("oS", "BASE", odovar);
    doinitilm = lower_typeload(dtype, ilm);
    ilm = plower("oS", "BASE", dovar);
    lower_typestore(dtype, ilm, doinitilm);

    /* compute trip count if it is not chunk 1 */
    /* lpct = ((lb-ub)+doinc)/doinc */
    if (!chunkone) {
      doendilm = plower("oS", "BASE", newend);
      doendilm = lower_typeload(dtype, doendilm);
      doincilm = plower("oS", "BASE", doinc);
      doincilm = lower_typeload(dtype, doincilm);
      dotripilm = compute_dotrip(std, 0, doinitilm, doendilm, doinc, doincilm,
                                 dtype, dotrip);
    }

    /* If this is distributed loop and chunk size is not specified
     * loop get splitted among teams as a block scheduling.
     * We don't need to loop back to get next chunk.  We can
     * just split the loop among teams and return.
     */

    if (is_dist == 1) {
      /* distribute parallel do, don't do trip - let parallel do handle it */
      itop = lower_lab();
      ibottom = lower_lab();
      dotrip = 0;
    } else {
      /* DOBEG(dotrip, lab, lab) */
      itop = lower_lab();
      ibottom = lower_lab();
      if (!chunkone) {
        plower_pdo(itop, schedtype);
        ilm = lower_typeload(DT_INT4, ilm);
        plower("oisS", "DOBEG", ilm, ibottom, dotrip);
      }
    }

    plower("oL", "LABEL", itop);

    /* check chunk one, thread 1 */
    if (chunkone) {
      check_loop_bound(dovar, newend, newend, dost, ibottom, 0, incr_loop);
    }

    if (is_dist == 1) {
      int ilm, initsptr, endsptr, incsptr;

      initsptr = A_SPTRG(A_M1G(ast));
      ilm = plower("oS", "BASE", dovar);
      ilm = lower_typeload(dtype, ilm);
      set_mp_loop_var(initsptr, ilm, dtype);

      endsptr = A_SPTRG(A_M2G(ast));
      ilm = plower("oS", "BASE", newend);
      ilm = lower_typeload(dtype, ilm);
      set_mp_loop_var(endsptr, ilm, dtype);

      incsptr = A_SPTRG(doincilm);
    }

    lower_end_stmt(std);
  }

  lower_push(0); /* no 'dotrip' variable */
  lower_push(dotop);
  lower_push(dobottom);
  lower_push(newend);    /* used by openmp llvm */
  lower_push(odovar);    /* no do variable */
  lower_push(dost);      /* no 'doinc' variable */
  lower_push(schedtype); /* schedtype */
  lower_push(STKDO);

  lower_push(dotrip); /* no 'dotrip' variable */
  lower_push(itop);
  lower_push(ibottom);
  lower_push(0);
  lower_push(dovar); /* no 'dovar' variable */
  if (dyn == 1)
    lower_push(dost); /* no 'dost' variable */
  else
    lower_push(doinc);   /* no 'doinc' variable */
  lower_push(schedtype); /* schedtype */
  lower_push(STKDO);
}

/*
 * Handle DO statements, and OpenMP PDO statements.
 * For PDO, if schedule type is STATIC and a chunk size is given,
 * use cyclic (if chunk size is one) or block-cyclic (otherwise)
 * scheduling;  otherwise, use static block scheduling (for now)
 */
#ifdef FLANG_LOWERILM_UNUSED
static int lcpu2(int);
static int ncpus2(int);
#endif

static void
lower_do_stmt(int std, int ast, int lineno, int label)
{
  int doinitast, doendast, doincast, plast;
  int dotop, dobottom, dotrip, doinc, dovar;
  int doinitilm, doendilm, doincilm, dotripilm, lop, lilm, ilm;
  int dtype, schedtype;
  int hack;

  plast = A_LASTVALG(ast);
  if (!plast) {
    plast = stb.i0;
  } else {
    plast = A_SPTRG(plast);
  }

  lower_start_stmt(lineno, label, TRUE, std);
  /* need two labels, for loop top and zero-trip exit.
   * need a temporary to hold trip count */
  dotop = lower_lab();
  if (STD_BLKSYM(std))
    STARTLABP(STD_BLKSYM(std), dotop); // overwrite any non-innermost loop label
  dobottom = lower_lab();
  ++lowersym.docount;
  lop = A_DOVARG(ast);
  if (A_TYPEG(lop) != A_ID) {
    lerror("unsupported DO variable");
    return;
  }
  dovar = A_SPTRG(lop);
  dtype = DTYPEG(dovar);
  /* treat logical like integer */
  switch (dtype) {
  case DT_BLOG:
    dtype = DT_BINT;
    break;
  case DT_SLOG:
    dtype = DT_SINT;
    break;
  case DT_LOG4:
    dtype = DT_INT4;
    break;
  case DT_LOG8:
    dtype = DT_INT8;
    break;
  }
  /* KMPC only permits 4 or 8 byte loop inductions */
  if (A_TYPEG(ast) == A_MP_PDO)
    dtype = (size_of(dtype) <= 4) ? DT_INT : DT_INT8;
  if (dtype == DT_INT8 && (XBIT(49, 0x100) || XBIT(68, 0x1)))
    dotrip = dotemp(STD_BLKSYM(std)?'C':'Y', DT_INT8, std);
  else
    dotrip = dotemp(STD_BLKSYM(std)?'C':'Y', DT_INT4, std);
  PTRSAFEP(dotrip, 1);
  doinitast = A_M1G(ast);
  doendast = A_M2G(ast);
  doincast = A_M3G(ast);
  /* compute global initial value */
  lower_expression(doinitast);
  doinitilm = lower_ilm(doinitast);
  doinitilm = lower_conv_ilm(doinitast, doinitilm, A_NDTYPEG(doinitast), dtype);
  lower_reinit();
  /* compute global increment */
  doinc = doincilm = 0;
  if (doincast == 0) {
    /* implicit increment is one */
    switch (DTY(DTYPEG(dovar))) {
    case TY_INT8:
    case TY_LOG8:
      doinc = lowersym.bnd.one;
      doincast = astb.bnd.one;
      break;
    case TY_BINT:
    case TY_SINT:
    case TY_INT:
    case TY_BLOG:
    case TY_SLOG:
    case TY_LOG:
      doinc = stb.i1;
      doincast = astb.i1;
      break;
    case TY_REAL:
      doinc = stb.flt1;
      doincast = mk_cnst(stb.flt1);
      break;
    case TY_DBLE:
      doinc = stb.dbl1;
      doincast = mk_cnst(stb.dbl1);
      break;
    default:
      ast_error("unexpected type for do variable", ast);
      doinc = stb.i1;
      doincast = astb.i1;
      break;
    }
  } else if (A_ALIASG(doincast)) {
    doincast = A_ALIASG(doincast);
    doinc = A_SPTRG(doincast);
    if (DTYPEG(doinc) != dtype) {
      lower_expression(doincast);
      doincilm = lower_ilm(doincast);
      doincilm = lower_conv_ilm(doincast, doincilm, A_NDTYPEG(doincast), dtype);
      lower_reinit();
    }
  } else {
    if (doincast == doinitast) {
      doincilm = doinitilm;
    } else {
      lower_expression(doincast);
      doincilm = lower_ilm(doincast);
      doincilm = lower_conv_ilm(doincast, doincilm, A_NDTYPEG(doincast), dtype);
      lower_reinit();
    }
  }
  lower_expression(doendast);
  ilm = lower_ilm(doendast);
  doendilm = lower_conv_ilm(doendast, ilm, A_NDTYPEG(doendast), dtype);

  if (A_TYPEG(ast) != A_MP_PDO || A_TASKLOOPG(ast)) {
    /* sequential DO:
     *  doinc = doincilm
     *  dovar = doinitilm
     *  DOBEG(dotrip,lab,lab)
     *   ...
     *  dovar = dovar + doinc
     *  DOEND(lab,lab)
     */
    if (A_TASKLOOPG(ast)) {
      /* lower taskloop as a regular loop */
      int ub;
      if (doinc == 0) {
        /* convert and store in a temp */
        doinc = dotemp('i', dtype, std);
        lilm = lower_sptr(doinc, VarBase);
        lower_typestore(dtype, lilm, doincilm);
      }
      ub = dotemp('U', dtype, std);
      plower("ossssd", "MPTASKLOOP", dovar, ub, doinc, plast, dtype);

      /* those values will be loaded from task alloc at the
       * beginning of an outlined function.
       */
      ilm = plower("oS", "BASE", dovar);
      doinitilm = lower_typeload(DTYPEG(dovar), ilm);
      ilm = plower("oS", "BASE", ub);
      doendilm = lower_typeload(DTYPEG(ub), ilm);
      ilm = plower("oS", "BASE", doinc);
      doincilm = lower_typeload(DTYPEG(doinc), ilm);
      ilm = compute_dotrip(std, FALSE, doinitilm, doendilm, doinc, doincilm,
                           dtype, dotrip);
    } else
    {
      ilm = compute_dotrip(std, doinitast == doincast, doinitilm, doendilm,
                           doinc, doincilm, dtype, dotrip);
      if (doinc == 0) {
        /* convert and store in a temp */
        doinc = dotemp('i', dtype, std);
        lilm = lower_sptr(doinc, VarBase);
        lower_typestore(dtype, lilm, doincilm);
      }
      lilm = lower_sptr(dovar, VarBase);
      lower_typestore(dtype, lilm, doinitilm);
    }
    if (!XBIT(34, 0x8000000) && STD_ZTRIP(std) && A_M4G(ast)) {
      /* lower condition ilm */
      int cilm = A_M4G(ast);
      if (cilm) {
        lower_expression(cilm);
        cilm = lower_ilm(cilm);
        lower_reinit();
      }
      plower("oisSi", "DOBEGNZ", ilm, dobottom, dotrip, cilm);
    } else {
      plower("oisS", "DOBEG", ilm, dobottom, dotrip);
    }
    plower("oL", "LABEL", dotop);
    lower_end_stmt(std);
    /* save labels, trip, inc, dovar info for ENDDO statement */
    schedtype = 0;
    lower_push(dotrip);
    lower_push(dotop);
    lower_push(dobottom);
    lower_push(0); /*  used by openmp llvm */
    lower_push(dovar);
    lower_push(doinc);
    lower_push(schedtype);
    lower_push(STKDO);
  } else if (A_ORDEREDG(ast) || A_SCHED_TYPEG(ast) == DI_SCH_DYNAMIC ||
             A_SCHED_TYPEG(ast) == DI_SCH_GUIDED ||
             A_SCHED_TYPEG(ast) == DI_SCH_RUNTIME ||
             A_SCHED_TYPEG(ast) == DI_SCH_AUTO) {
    /* call runtime routine.
     *  _mp_scheds_[static|dyn|guid|run]_init[|8](
     *              int(32+MAXCPUS), int(32), %val(doinitilm),
     *		%val(dotripilm), %val(doincilm),
     *		%val(chunkilm) )
     *  while( _mp_scheds( int(32+MAXCPUS), int(32), dovar, dotripvar ) )
     *   DOBEG( dotripvar, , )
     *   PDO
     *    ...
     *    dovar = dovar + step
     *   DOEND(lab,lab)
     *  DOEND(lab,lab)
     */
    int dosched, chunkast, chunkilm, schedfunc, v[6], doend;
    int doscheds;
    char schedname[40];

    switch (A_SCHED_TYPEG(ast)) {
    case DI_SCH_STATIC:
      strcpy(schedname, "_mp_scheds_static_init");
      if (A_CHUNKG(ast) == 0)
        A_CHUNKP(ast, astb.i0);
      break;
    case DI_SCH_DYNAMIC:
      strcpy(schedname, "_mp_scheds_dyn_init");
      break;
    case DI_SCH_GUIDED:
      strcpy(schedname, "_mp_scheds_guid_init");
      break;
    case DI_SCH_RUNTIME:
      strcpy(schedname, "_mp_scheds_run_init");
      break;
    case DI_SCH_AUTO:
      strcpy(schedname, "_mp_scheds_auto_init");
      break;
    }
    if (dtype == DT_INT8) {
      strcat(schedname, "8");
    }

    if (dtype != DT_INT8)
      hack = dotrip;
    else
      hack = dotemp('e', DT_INT8, std);

    set_mp_loop_var(dovar, doinitilm, dtype);
    set_mp_loop_var(hack, doendilm, dtype);

    if (lowersym.sched_dtype == 0) {
      int dt;
      /* create datatype for schedule function argument */
      dt = get_array_dtype(1, DT_INT);
      lower_use_datatype(DT_INT, 1);
      ADD_ZBASE(dt) = astb.bnd.one;
      ADD_MLPYR(dt, 0) = astb.bnd.one;
      ADD_LWBD(dt, 0) = ADD_LWAST(dt, 0) = astb.bnd.one;
      ADD_NUMELM(dt) = ADD_UPBD(dt, 0) = ADD_UPAST(dt, 0) =
          ADD_EXTNTAST(dt, 0) = mk_cnst(lower_getiszcon(32));
      lowersym.sched_dtype = dt;
    }
    if (lowersym.scheds_dtype == 0) {
      int dt;
      /* create datatype for schedule function argument */
      dt = get_array_dtype(1, DT_INT);
      lower_use_datatype(DT_INT, 1);
      ADD_ZBASE(dt) = astb.bnd.one;
      ADD_MLPYR(dt, 0) = astb.bnd.one;
      ADD_LWBD(dt, 0) = ADD_LWAST(dt, 0) = astb.bnd.one;
      ADD_NUMELM(dt) = ADD_UPBD(dt, 0) = ADD_UPAST(dt, 0) =
          ADD_EXTNTAST(dt, 0) = mk_cnst(lower_getiszcon(32 + MAXCPUS));
      lowersym.scheds_dtype = dt;
    }
    /* make scheduler data structure */
    dosched = dotemp('r', lowersym.sched_dtype, std);
    doscheds = getccssym_sc("yR", lowersym.docount, ST_ARRAY, SC_LOCAL);
    DTYPEP(doscheds, lowersym.scheds_dtype);
    SAVEP(doscheds, 1);

    /* get chunk size */
    chunkast = A_CHUNKG(ast);
    if (chunkast == 0) {
      if (dtype == DT_INT8)
        chunkast = astb.k1;
      else
        chunkast = astb.i1;
    }
    lower_expression(chunkast);
    chunkilm = lower_conv(chunkast, dtype);

    /* doinc = doincilm */
    if (doinc == 0) {
      /* convert and store in a temp */
      doinc = dotemp('i', dtype, std);
      ilm = lower_sptr(doinc, VarBase);
      lower_typestore(dtype, ilm, doincilm);
    } else if (STYPEG(doinc) == ST_CONST) {
      doincilm = plower("oS", ltyped("CON", dtype), doinc);
    }
    if (A_SCHED_TYPEG(ast) == DI_SCH_STATIC)
      schedtype = 0x0;
    else if (A_SCHED_TYPEG(ast) == DI_SCH_RUNTIME)
      schedtype = 0x4;
    else if (A_SCHED_TYPEG(ast) == DI_SCH_GUIDED)
      schedtype = 0x2;
    else if (A_SCHED_TYPEG(ast) == DI_SCH_AUTO)
      schedtype = 0x5;
    else
      schedtype = 0x1;
    if (A_ORDEREDG(ast)) {
      if ((A_SCHED_TYPEG(ast) == DI_SCH_AUTO) ||
          (A_SCHED_TYPEG(ast) == DI_SCH_RUNTIME)) {
        schedtype = schedtype | MP_SCH_ATTR_ORDERED;
      } else {
        int chk = MP_SCH_ATTR_CHUNKED;
        /* all dynamic are chunk for kmpc */
        if (A_CHUNKG(ast) == astb.i0 && A_SCHED_TYPEG(ast) == DI_SCH_STATIC)
          chk = 0;
        schedtype = schedtype | MP_SCH_ATTR_ORDERED | chk;
      }
    }
    llvm_omp_sched(std, ast, dtype, dotop, dobottom, dovar, plast, dotrip,
                   doinitilm, doinc, doincilm, doendilm, schedtype, lineno);
    return;

    /* If ordered, call the order_init function -
     *     _mp_orders_init(its_sym, from, to, inc);
     */
    if (A_ORDEREDG(ast)) {
      strcpy(schedname, "_mp_orders_init");
      if (dtype == DT_INT8)
        strcat(schedname, "8");
      schedfunc = lower_makefunc(schedname, DT_NONE, TRUE);
      /*
      v[2] = plower( "oi", "DPVAL", doinitilm );
      v[3] = plower( "oi", "DPVAL", doendilm );
      v[4] = plower( "oi", "DPVAL", doincilm );
      */
      plower("onsiiiiC", "CALL", 4, schedfunc, v[0], v[2], v[3], v[4],
             schedfunc);
    }
    lower_end_stmt(0);
    lower_start_stmt(lineno, 0, TRUE, 0);

    /* dowhile( _mp_scheds( struct, struct,&from,&to ) ) */
    plower("oL", "LABEL", dotop);
    strcpy(schedname, "_mp_scheds");
    if (dtype == DT_INT8) {
      strcat(schedname, "8");
    }
    schedfunc = lower_makefunc(schedname, DT_INT4, TRUE);

    doend = dotemp('e', DT_INT4, std);
    if (dtype != DT_INT8)
      hack = dotrip;
    else
      hack = dotemp('e', DT_INT8, std);
    v[0] = plower("oS", "BASE", doscheds);
    v[1] = plower("oS", "BASE", dosched);
    v[2] = plower("oS", "BASE", dovar);
    v[3] = plower("oS", "BASE", /*dotrip*/ hack);
    ilm = plower("onsiiiiC", ltyped("FUNC", DT_INT4), 4, schedfunc, v[0], v[1],
                 v[2], v[3], schedfunc);
    lilm = plower("oS", "ICON", lowersym.intzero);
    ilm = plower("oii", "ICMP", ilm, lilm);
    ilm = plower("oi", "EQ", ilm);
    plower("oiS", "BRT", ilm, dobottom);

    /* set chunk if ordered
     *     _mp_orders_chunk(from, to);
     */
    if (A_ORDEREDG(ast)) {
      strcpy(schedname, "_mp_orders_chunk");
      if (dtype == DT_INT8)
        strcat(schedname, "8");
      schedfunc = lower_makefunc(schedname, DT_NONE, TRUE);
      v[2] = lower_typeload(dtype, v[2]);
      v[2] = plower("oi", "DPVAL", v[2]);
      v[3] = lower_typeload(dtype, v[3]);
      v[3] = plower("oi", "DPVAL", v[3]);
      plower("onsiiC", "CALL", 2, schedfunc, v[2], v[3], schedfunc);
    }

    lower_end_stmt(0);
    lower_start_stmt(lineno, 0, TRUE, 0);
    lower_push(0); /* no 'dotrip' variable */
    lower_push(dotop);
    lower_push(dobottom);
    lower_push(0); /*  used by openmp llvm */
    lower_push(0); /* no 'dovar' variable */
    lower_push(0); /* no 'doinc' variable */
    lower_push(0); /* schedtype */
    lower_push(STKDO);

    dotop = lower_lab();
    dobottom = lower_lab();

    ilm = plower("oS", "BASE", dovar);
    doinitilm = lower_typeload(dtype, ilm);
    ilm = plower("oS", "BASE", /*dotrip*/ hack);
    doendilm = lower_typeload(dtype, ilm);
    if (STYPEG(doinc) != ST_CONST) {
      ilm = plower("oS", "BASE", doinc);
      doincilm = lower_typeload(dtype, ilm);
    }

    dotripilm = compute_dotrip(std, 0, doinitilm, doendilm, doinc, doincilm,
                               dtype, dotrip);

    plower("oisS", "DOBEG", dotripilm, dobottom, dotrip);
    plower("oL", "LABEL", dotop);
    schedtype = A_SCHED_TYPEG(ast);
    plower_pdo(dotop, schedtype);
    lower_end_stmt(std);
/* save labels, trip, inc, dovar info for ENDDO statement */
#define ORDERED_MASK 0x10000
    if (A_ORDEREDG(ast))
      schedtype = ORDERED_MASK;
    lower_push(dotrip);
    lower_push(dotop);
    lower_push(dobottom);
    lower_push(0); /*  used by openmp llvm */
    lower_push(dovar);
    lower_push(doinc);
    lower_push(schedtype);
    lower_push(STKDO);
  } else if (A_CHUNKG(ast) == astb.i1) {
    /* cyclic scheduling:
     *  doinitilm = doinitilm + doincilm*lcpu
     *  doinc = doincilm = doincilm*ncpus
     *  trip = (doendilm-doinitilm+doincilm)/doincilm
     *  dovar = doinitilm
     *  DOBEG(trip,lab,lab)
     *  PDO
     *   ...
     *  dovar = dovar + step
     *  DOEND(lab,lab)
     */
    schedtype = (MP_SCH_ATTR_CHUNKED | MP_SCH_CHUNK_1);
    if (A_SCHED_TYPEG(ast) == MP_SCH_DIST_STATIC) {
      schedtype = schedtype | MP_SCH_DIST_STATIC;
    }
    llvm_omp_sched(std, ast, dtype, dotop, dobottom, dovar, plast, dotrip,
                   doinitilm, doinc, doincilm, doendilm, schedtype, lineno);
  } else if (A_CHUNKG(ast) == 0) {
    /* block scheduling:
     *  dovar = doinitilm
     *  doinc = doincilm
     *     ... [no]aligned setup ...
     *  DOBEG(ldotrip,lab,lab)
     *  PDO
     *   ...
     *  dovar = dovar + step
     *  DOEND(lab,lab)
     */
    schedtype = 0x000;
    if (A_SCHED_TYPEG(ast) == MP_SCH_DIST_STATIC) {
      schedtype = MP_SCH_DIST_STATIC;
    }
    llvm_omp_sched(std, ast, dtype, dotop, dobottom, dovar, plast, dotrip,
                   doinitilm, doinc, doincilm, doendilm, schedtype, lineno);
  } else {
    /* block-cyclic scheduling:
     *  oinit = ginit + step*chunk*lcpu
     *  ostep = gstep**chunk*ncpus
     *  otrip = (gfinal-oinit+ostep)/ostep
     *  odovar = oinit
     *  DOBEG(otrip,lab,lab)
     *  PDO
     *  iinit = oinit
     *  istep = gstep
     *  itrip = min(chunk,(gfinal-iinit+istep)/istep)
     *  dovar = iinit
     *  DOBEG(itrip,lab,lab)
     *   ...
     *  dovar = dovar + step
     *  DOEND(lab,lab)
     *  odovar = odovar + ostep
     *  DOEND(lab,lab)
     */
    schedtype = (MP_SCH_ATTR_CHUNKED | MP_SCH_BLK_CYC);
    if (A_SCHED_TYPEG(ast) == MP_SCH_DIST_STATIC) {
      schedtype = schedtype | MP_SCH_DIST_STATIC;
    }
    llvm_omp_sched(std, ast, dtype, dotop, dobottom, dovar, plast, dotrip,
                   doinitilm, doinc, doincilm, doendilm, schedtype, lineno);
  }
} /* lower_do_stmt */

#ifdef FLANG_LOWERILM_UNUSED
static int
lcpu2(int dt)
{
  int ilm;
  ilm = plower("o", "LCPU2");
  if (dt == DT_INT8)
    ilm = plower("oi", "ITOI8", ilm);
  return ilm;
}
#endif

#ifdef FLANG_LOWERILM_UNUSED
static int
ncpus2(int dt)
{
  int ilm;
  ilm = plower("o", "NCPUS2");
  if (dt == DT_INT8)
    ilm = plower("oi", "ITOI8", ilm);
  return ilm;
}
#endif

static void
llvm_lower_enddo_stmt(int lineno, int label, int std, int ispdo)
{
  int dotop, dobottom, doinc, dotrip, dovar, doub, docancel;
  int dtype, fdtype, schedtype;
  int lilm, rilm, ilm, dyn, tmpsched;
  int chunkone = 0;
  int block_cyclic = 0;

  int idotop, idobottom, idoinc, idotrip, idovar, idoub;
  int ischedtype;

  lower_start_stmt(lineno, label, TRUE, std);

  /* inner loop - do loop */
  lower_check_stack(STKDO);
  ischedtype = lower_pop(); /* schedule type */
  idoinc = lower_pop();     /* do increment symbol */
  idovar = lower_pop();     /* do variable */
  idoub = lower_pop();      /* label used for chunk 1  */
  idobottom = lower_pop();  /* do bottom label */
  idotop = lower_pop();     /* do top label */
  idotrip = lower_pop();    /* do trip count variable */

  /* outer loop - do while */
  lower_check_stack(STKDO);
  schedtype = lower_pop(); /* schedule type */
  doinc = lower_pop();     /* do increment symbol */
  dovar = lower_pop();     /* do variable */
  doub = lower_pop();      /* do upperbound symbol - openmp llvm used  */
  dobottom = lower_pop();  /* do bottom label */
  dotop = lower_pop();     /* do top label */
  dotrip = lower_pop();    /* do trip count variable */

  /* cancel/cancellation */
  lower_check_stack(STKCANCEL);
  docancel = lower_pop();

  dyn = 0;
  tmpsched = (schedtype & MP_SCH_TYPE_MASK);
  if ((tmpsched == MP_SCH_DYNAMIC) || (tmpsched == MP_SCH_GUIDED) ||
      (tmpsched == MP_SCH_RUNTIME) || (tmpsched == MP_SCH_AUTO) ||
      (schedtype & MP_SCH_ATTR_ORDERED)) {
    dyn = 1;
  } else if (schedtype & MP_SCH_CHUNK_1) {
    chunkone = 1;
  } else if (schedtype & MP_SCH_BLK_CYC) {
    block_cyclic = 1;
  }

  if (idovar) {
    dtype = DTYPEG(idovar);
    fdtype = dtype;
    switch (DTY(dtype)) {
    case TY_BLOG:
      fdtype = DT_BINT;
      break;
    case TY_SLOG:
      fdtype = DT_SINT;
      break;
    case TY_LOG:
      fdtype = DT_INT;
      break;
    case TY_LOG8:
      fdtype = DT_INT8;
      break;
    }
  }
  /* increment the do variable */
  if (idotrip) {
    /* DO loop */
    if ((schedtype & MP_SCH_ATTR_ORDERED)) {
      plower("odn", "MPLOOPFINI", dtype, schedtype);
    }
    lilm = lower_sptr(idovar, VarBase);
    lilm = lower_typeload(dtype, lilm);
    if (STYPEG(idoinc) == ST_VAR) {
      rilm = lower_sptr(idoinc, VarBase);
      rilm = lower_typeload(dtype, rilm);
    } else {
      rilm = plower("oS", ltyped("CON", fdtype), idoinc);
    }
    ilm = plower("oii", ltyped("ADD", fdtype), lilm, rilm);
    lilm = lower_sptr(idovar, VarBase);
    lower_typestore(dtype, lilm, ilm);
    if (!chunkone) {
      if (!XBIT(34, 0x8000000) && STD_ZTRIP(std))
        plower("oss", "DOENDNZ", idotop, idotrip);
      else
        plower("oss", "DOEND", idotop, idotrip);
    } else {
      plower("oS", "BR", idotop);
      lower_end_stmt(std);
      lower_start_stmt(lineno, label, TRUE, std);
    }
  }
  plower("oL", "LABEL", idobottom);

  lower_end_stmt(std);

  /* do while loop */
  if ((ischedtype != 0) && (ischedtype != MP_SCH_DIST_STATIC)) {
    lower_start_stmt(lineno, label, TRUE, std);
    if (dyn == 0) {
      lilm = lower_sptr(dovar, VarBase);
      ilm = lower_typeload(dtype, lilm);
      rilm = lower_sptr(doinc, VarBase);
      rilm = lower_typeload(dtype, rilm);
      ilm = plower("oii", ltyped("ADD", dtype), rilm, ilm);
      lower_typestore(dtype, lilm, ilm);

      lilm = lower_sptr(doub, VarBase);
      ilm = lower_typeload(dtype, lilm);
      ilm = plower("oii", ltyped("ADD", dtype), rilm, ilm);
      lower_typestore(dtype, lilm, ilm);
    }

    plower("oS", "BR", dotop);
    if (docancel) {
      plower("oL", "LABEL", docancel);
    }
    plower("oL", "LABEL", dobottom);
    if (!(schedtype & MP_SCH_ATTR_ORDERED)) {
      plower("odn", "MPLOOPFINI", dtype, schedtype);
    }
    lower_end_stmt(std);

  } else if (dobottom) {
    lower_start_stmt(lineno, label, TRUE, std);
    if (docancel) {
      plower("oL", "LABEL", docancel);
    }
    plower("oL", "LABEL", dobottom);

    plower("odn", "MPLOOPFINI", dtype, schedtype);
    lower_end_stmt(std);
  }

} /* llvm_lower_enddo_stmt */

static void
lower_enddo_stmt(int lineno, int label, int std, int ispdo)
{
  int dotop, dobottom, doinc, dotrip, dovar, doub;
  int dtype, fdtype, schedtype;
  int lilm, rilm, ilm;
  if (ispdo) {
    llvm_lower_enddo_stmt(lineno, label, std, ispdo);
    return;
  }
  lower_start_stmt(lineno, label, TRUE, std);
  lower_check_stack(STKDO);
  schedtype = lower_pop(); /* schedule type */
  doinc = lower_pop();     /* do increment symbol */
  dovar = lower_pop();     /* do variable */
  doub = lower_pop();      /* do upperbound symbol - openmp llvm used  */
  dobottom = lower_pop();  /* do bottom label */
  dotop = lower_pop();     /* do top label */
  dotrip = lower_pop();    /* do trip count variable */
  if (dovar) {
    dtype = DTYPEG(dovar);
    fdtype = dtype;
    switch (DTY(dtype)) {
    case TY_BLOG:
      fdtype = DT_BINT;
      break;
    case TY_SLOG:
      fdtype = DT_SINT;
      break;
    case TY_LOG:
      fdtype = DT_INT;
      break;
    case TY_LOG8:
      fdtype = DT_INT8;
      break;
    }
  }
  /* increment the do variable */
  if (dotrip) {
    /* DO loop */
    lilm = lower_sptr(dovar, VarBase);
    lilm = lower_typeload(dtype, lilm);
    if (STYPEG(doinc) == ST_VAR) {
      rilm = lower_sptr(doinc, VarBase);
      rilm = lower_typeload(dtype, rilm);
    } else {
      rilm = plower("oS", ltyped("CON", fdtype), doinc);
    }
    ilm = plower("oii", ltyped("ADD", fdtype), lilm, rilm);
    lilm = lower_sptr(dovar, VarBase);
    lower_typestore(dtype, lilm, ilm);
    if (!XBIT(34, 0x8000000) && STD_ZTRIP(std))
      plower("oss", "DOENDNZ", dotop, dotrip);
    else
      plower("oss", "DOEND", dotop, dotrip);
  } else {
/* DOWHILE loop */
    if (!ispdo)
      plower("oS", "BR", dotop);
  }
  if (!(ispdo && schedtype & 0x200))
    plower("oL", "LABEL", dobottom);
  if (schedtype & 0x100) {
    /* cyclic scheduling */
    int oneilm;
    lilm = lower_sptr(dovar, VarBase);
    lilm = lower_typeload(dtype, lilm);
    if (STYPEG(doinc) == ST_VAR) {
      rilm = lower_sptr(doinc, VarBase);
      rilm = lower_typeload(dtype, rilm);
    } else {
      rilm = plower("oS", ltyped("CON", fdtype), doinc);
    }
    ilm = plower("oii", ltyped("SUB", fdtype), lilm, rilm);
    oneilm = plower("oS", "ICON", lowersym.intone);
    if (dtype == DT_INT8)
      oneilm = plower("oi", "ITOI8", oneilm);
    ilm = plower("oii", ltyped("ADD", dtype), oneilm, ilm);
    lilm = lower_sptr(dovar, VarBase);
    lower_typestore(dtype, lilm, ilm);
    lower_end_stmt(std);
  } else if (schedtype & 0x200) {
    /* block-cyclic PDO loop */
    lower_end_stmt(0);
    lower_enddo_stmt(lineno, 0, std, ispdo);

    if (ispdo && !dotrip) {
      lower_start_stmt(lineno, label, TRUE, std);

      lilm = lower_sptr(dovar, VarBase);
      ilm = lower_typeload(dtype, lilm);
      rilm = lower_sptr(doinc, VarBase);
      rilm = lower_typeload(dtype, rilm);
      ilm = plower("oii", ltyped("ADD", dtype), rilm, ilm);
      lower_typestore(dtype, lilm, ilm);

      lilm = lower_sptr(doub, VarBase);
      ilm = lower_typeload(dtype, lilm);
      ilm = plower("oii", ltyped("ADD", dtype), rilm, ilm);
      lower_typestore(dtype, lilm, ilm);

      plower("oS", "BR", dotop);
      plower("oL", "LABEL", dobottom);
      plower("odn", "MPLOOPFINI", dtype, schedtype);
      lower_end_stmt(std);

      /* if lastprivate */
      lower_start_stmt(lineno, label, TRUE, std);
      lilm = lower_sptr(dovar, VarBase);
      ilm = lower_typeload(dtype, lilm);
      rilm = lower_sptr(doinc, VarBase);
      rilm = lower_typeload(dtype, rilm);
      ilm = plower("oii", ltyped("SUB", dtype), ilm, rilm);
      lower_typestore(dtype, lilm, ilm);
      lower_end_stmt(std);
    }
  } else if ((schedtype & ORDERED_MASK) || schedtype == DI_SCH_DYNAMIC ||
             schedtype == DI_SCH_GUIDED || schedtype == DI_SCH_RUNTIME ||
             schedtype == DI_SCH_AUTO
             || (schedtype & MP_SCH_ATTR_ORDERED)
  ) {
/* handle the 'while' loop */
    if (ispdo && (schedtype & MP_SCH_ATTR_ORDERED)) {
      plower("odn", "MPLOOPFINI", dtype, schedtype);
    }
    lower_end_stmt(0);
    lower_enddo_stmt(lineno, 0, std, ispdo);
  } else {
    lower_end_stmt(std);
  }

} /* lower_enddo_stmt */

#ifdef FLANG_LOWERILM_UNUSED
static void
lower_omp_atomic_read(int ast, int lineno)
{
  int sptr, rilm;
  int src;
  int mem_order;

  src = A_SRCG(ast);
  mem_order = A_MEM_ORDERG(ast);

  /* lower rhs and convert type to lhs type */
  lower_expression(src);
  rilm = lower_base(src);
  if (!NDTYPE_IS_SET(ast)) {
    A_NDTYPEP(ast, A_DTYPEG(ast));
  }
  plower("oin", "MP_ATOMICREAD", rilm, mem_order);
}
#endif

static void
lower_omp_atomic_write(int ast, int lineno)
{
  int lilm, rilm;
  int lop, rop;
  int mem_order;

  lop = A_LOPG(ast);
  rop = A_ROPG(ast);
  mem_order = A_MEM_ORDERG(ast);

  lower_expression(lop);
  lilm = lower_base(lop);

  /* lower rhs and convert type to lhs type */
  lower_expression(rop);
  rilm = lower_conv(rop, A_DTYPEG(lop));

  plower("oiin", "MP_ATOMICWRITE", lilm, rilm, mem_order);
}

void static lower_omp_atomic_update(int ast, int lineno)
{
  int lilm, rilm;
  int lop, rop;
  int mem_order;
  int aop;

  lop = A_LOPG(ast);
  rop = A_ROPG(ast);
  mem_order = A_MEM_ORDERG(ast);
  aop = A_OPTYPEG(ast);

  lower_expression(lop);
  lilm = lower_base(lop);

  /* lower rhs and convert type to lhs type */
  lower_expression(rop);
  rilm = lower_conv(rop, A_DTYPEG(lop));

  plower("oiinn", "MP_ATOMICUPDATE", lilm, rilm, mem_order, aop);
}

static void
lower_omp_atomic_capture(int ast, int lineno)
{
  int lilm, rilm;
  int lop, rop;
  int aop;
  int mem_order;
  int flag = 0;

  lop = A_LOPG(ast);
  rop = A_ROPG(ast);
  mem_order = A_MEM_ORDERG(ast);
  aop = A_OPTYPEG(ast);

  lower_expression(lop);
  lilm = lower_base(lop);

  /* lower rhs and convert type to lhs type */
  lower_expression(rop);
  rilm = lower_conv(rop, A_DTYPEG(lop));

  plower("oiinnn", "MP_ATOMICCAPTURE", lilm, rilm, mem_order, aop, flag);
}

static void
lower_omp_target_tripcount(int ast, int std)
{
  int lop,dovar,doinitast,doendast,dtype,doincast, doinc, doinitilm, doendilm, doincilm, dotrip;
  lop = A_DOVARG(ast);

  if (A_TYPEG(lop) != A_ID) {
    lerror("unsupported DO variable");
    return;
  }
  dovar = A_SPTRG(lop);
  dtype = DTYPEG(dovar);
  /* treat logical like integer */
  switch (dtype) {
    case DT_BLOG:
      dtype = DT_BINT;
      break;
    case DT_SLOG:
      dtype = DT_SINT;
      break;
    case DT_LOG4:
      dtype = DT_INT4;
      break;
    case DT_LOG8:
      dtype = DT_INT8;
      break;
  }
  /* KMPC only permits 4 or 8 byte loop inductions */
  if (A_TYPEG(ast) == A_MP_PDO)
    dtype = (size_of(dtype) <= 4) ? DT_INT : DT_INT8;
  if (XBIT(68, 0x1)) {
    if (dtype == DT_INT8)
      dotrip = dotemp('T', DT_INT8, std);
    else
      dotrip = dotemp('T', DT_INT4, std);
  } else {
    if (XBIT(49, 0x100) && dtype == DT_INT8)
      dotrip = dotemp('T', DT_INT8, std);
    else
      dotrip = dotemp('T', DT_INT4, std);
  }
  PTRSAFEP(dotrip, 1);
  doinitast = A_M1G(ast);
  doendast = A_M2G(ast);
  doincast = A_M3G(ast);

  lower_expression(doinitast);
  doinitilm = lower_ilm(doinitast);
  lower_expression(doendast);
  doendilm = lower_ilm(doendast);
  lower_expression(doincast);
  doincilm = lower_ilm(doincast);

  doinc = dotemp('i', dtype, std);

  compute_dotrip(std, FALSE, doinitilm, doendilm,
                             doinc, doincilm, dtype, dotrip);

  plower("oS", "MP_TARGETLOOPTRIPCOUNT", dotrip);
  return;
}

void
lower_stmt(int std, int ast, int lineno, int label)
{
  int dtype, lop, rop, lilm, rilm, ilm = 0, ilm2 = 0, asd, silm;
  int ndim, i, sptr, args, count, lab, nlab, fmt, labnum, tyilm, nullilm;
  int astli, src, nextstd, dest, ilm3, ilm4;
  int dotop, dobottom, doit, stblk;
  int symfunc, symargs[30], num, sym, secnum;
  iflabeltype iflab;
  int alloc_func, dealloc_func;
  int have_ptr_alloc;
  int prev, flag, proc_bind;
  int is_assumeshp;
  int lite_alloc;
  int lop2;
  int src_dsc = 0;
  int src_dsc_ast = 0;
  FtnRtlEnum rtlRtn;

  if (STD_LINENO(std))
    lower_line = STD_LINENO(std);
  dtype = -99999;

  switch (A_TYPEG(ast)) {
  case A_NULL:
    break;

    /* ------------- statement AST types ------------- */

  case A_AGOTO:
    count = 0;
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli))
      ++count;
    lower_start_stmt(lineno, label, TRUE, std);
    lower_expression(A_LOPG(ast));
    ilm = lower_ilm(A_LOPG(ast));
    plower("onim", "AGOTO", count, ilm);
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      plower("sm", A_SPTRG(ASTLI_AST(astli)));
    }
    plower("e");
    lower_end_stmt(std);
    break;

  case A_ALLOC:
    lower_start_stmt(lineno, label, TRUE, std);
    if (!XBIT(57, 0x8000000))
      lite_alloc = 1;
    else
      lite_alloc = 0;
    if (A_TKNG(ast) == TK_ALLOCATE) {
      /* generate assignments to fill in bounds, zbase, ... */
      int src, object, sptr, dtype, eltype, silm, numilm, size, isarray,
          sizeilm, lilm, is_pinned, symhandle, errmsg, errilm, firstilm,
          aln, alnilm, src_dtype;

      src_dtype = A_DTYPEG(ast); /* set by sourced/typed allocation */

      src = A_SRCG(ast);
      sptr = sym_of_ast(src);
      lower_visit_symbol(sptr);
      isarray = 0;
      numilm = 0;
      is_pinned = 0;
      symhandle = 0;
      errmsg = A_M3G(ast);
      aln = A_ALIGNG(ast);
      switch (A_TYPEG(src)) {
      case A_SUBSCR:
        object = A_LOPG(src);
        sptr = find_pointer_variable(object);
        dtype = (src_dtype && CLASSG(sptr) && DTY(src_dtype) == TY_ARRAY) ? 
                src_dtype : DTYPEG(sptr);
        if (DTY(dtype) == TY_ARRAY)
          isarray = 1;
        if (isarray && (!ADJARRG(sptr) || RESULTG(sptr))) {
          if (!LNRZDG(sptr) || !XBIT(52, 4)) {
            put_adjarr_bounds(dtype, A_ASDG(src), &numilm);
          } else if (LNRZDG(sptr) &&
                     ((XBIT(52, 4) && ALLOCG(sptr)) || POINTERG(sptr) ||
                      ALIGNG(sptr) || DISTG(sptr))) {
            int dd;
            dd = DTY(dtype - 1);
            if (dd > 0) {
              lerror("unknown linearized datatype");
            } else {
              dd = -dd;
              put_adjarr_bounds(dd, A_ASDG(src), &numilm);
            }
          } else if (SDSCG(sptr) == 0 || NODESCG(sptr) ||
                     STYPEG(SDSCG(sptr)) == ST_PARAM) {
            int dd;
            dd = DTY(dtype - 1);
            if (dd > 0) {
              lerror("unknown linearized datatype");
            } else {
              dd = -dd;
              put_adjarr_bounds(dd, A_ASDG(src), &numilm);
            }
          }
        }
        break;
      case A_ID:
      case A_MEM:
        object = src;
        sptr = find_pointer_variable(object);
        dtype = (!src_dtype || !CLASSG(sptr)) ? DTYPEG(sptr) : src_dtype;
        if (DTY(dtype) == TY_ARRAY)
          isarray = 1;
        if (isarray && !POINTERG(sptr) && !ALLOCATTRG(sptr) && !DISTG(sptr) &&
            !ALIGNG(sptr) && (!ADJARRG(sptr) || RESULTG(sptr))) {
          if (!LNRZDG(sptr) || !XBIT(52, 4)) {
            put_adjarr_bounds(dtype, -1, &numilm);
          } else if (SDSCG(sptr) == 0 || NODESCG(sptr) ||
                     STYPEG(SDSCG(sptr)) == ST_PARAM) {
            int dd;
            dd = DTY(dtype - 1);
            if (dd > 0) {
              lerror("unknown linearized datatype");
            } else {
              dd = -dd;
              put_adjarr_bounds(dd, -1, &numilm);
            }
          }
        }
        break;
      default:
        ast_error("unexpected allocate argument", ast);
        return;
      }
      lower_visit_symbol(sptr);
      have_ptr_alloc = 0;
      if (ALLOCG(sptr)) {
/* RTE_alloc() checks if the ALLOCATABLE array is
 * already allocated.
 */
        {
          int src_sptr, src_dtype;
          switch (A_TYPEG(A_STARTG(ast))) {
          case A_ID:
          case A_LABEL:
          case A_ENTRY:
          case A_SUBSCR:
          case A_SUBSTR:
          case A_MEM:
            src_sptr = memsym_of_ast(A_STARTG(ast));
            src_dtype = DTYPEG(src_sptr);
            break;
          default:
            src_sptr = 0;
            src_dtype = A_DTYPEG(A_STARTG(ast));
          }
          if (A_STARTG(ast) && (CLASSG(src_sptr) || CLASSG(sptr))) {
            if (DTY(src_dtype) == TY_ARRAY)
              src_dtype = DTY(src_dtype + 1);
            if (DTY(src_dtype) == TY_DERIVED) {
              src_dsc_ast = 0;
              if (src_sptr && STYPEG(src_sptr) != ST_MEMBER) {
                if (!ALLOCDESCG(src_sptr) && !CLASSG(src_sptr)) {
                  src_dsc = get_static_type_descriptor(DTY(src_dtype + 3));
                } else {
                  src_dsc = get_type_descr_arg(gbl.currsub, src_sptr);
                }
              } else if (!src_sptr || !CLASSG(src_sptr)) {
                int dty = (!src_sptr) ? src_dtype : DTYPEG(src_sptr);
                src_dsc = get_static_type_descriptor(DTY(dty + 3));
              } else {
                int sdsc_mem = SYMLKG(src_sptr);
                if (sdsc_mem == MIDNUMG(src_sptr) || PTRVG(sdsc_mem)) {
                  sdsc_mem = SYMLKG(sdsc_mem);
                  if (PTRVG(sdsc_mem) || !DESCARRAYG(sdsc_mem))
                    sdsc_mem = SYMLKG(sdsc_mem);
                }
                src_dsc = sdsc_mem;
                if (UNLPOLYG(DTY(src_dtype + 3)) && SDSCG(src_sptr) &&
                    STYPEG(SDSCG(src_sptr)) == ST_MEMBER) {
                  /* TBD: Is the choice of descriptor really dependent
                   * on a class(*) src?
                   */
                  src_dsc_ast =
                      check_member(A_STARTG(ast), mk_id(SDSCG(src_sptr)));
                } else {
                  src_dsc_ast =
                      mk_member(A_PARENTG(A_STARTG(ast)), mk_id(src_dsc),
                                A_DTYPEG(A_PARENTG(A_STARTG(ast))));
                }

                A_NDTYPEP(src_dsc_ast, A_DTYPEG(src_dsc_ast));
              }

              lower_visit_symbol(src_dsc);
              lowersym.alloc = 0;
              if ((XBIT(54, 0x1) || CLASSG(sptr) || is_or_has_poly(sptr) ||
                   has_finalized_component(sptr) || has_layout_desc(sptr)) &&
                  is_or_has_derived_allo(sptr)) {
/* For -Mallocatable=03 where the object
 * is a derived type with an allocatable component
 * or with a nested or inherited allocatable component,
 * use calloc to make sure everything is zeroed out.
 * Otherwise, we may get UMRs on intrinsic assignments
 * from functions and structure constructors due to the
 * nested nature of the allocation.
 *
 * Also use calloc if object is derived type
 * declared CLASS, has a finalized component, or
 * has a polymorphic allocatable component.
 */
                  alloc_func = lower_makefunc(mkRteRtnNm(RTE_ptr_src_calloc04a),
                                              DT_NONE, FALSE);
              } else {
                  alloc_func = lower_makefunc(mkRteRtnNm(RTE_ptr_src_alloc04a),
                                              DT_NONE, FALSE);
              }
            } else {
              if (lowersym.alloc == 0) {
                  lowersym.alloc =
                      lower_makefunc(mkRteRtnNm(RTE_alloc04a), DT_NONE, FALSE);
              }
              if (lowersym.alloc_chk == 0) {
                  lowersym.alloc_chk = lower_makefunc(
                      mkRteRtnNm(RTE_alloc04_chka), DT_NONE, FALSE);
              }
              if (ALLOCATTRG(sptr)) {
                alloc_func = lowersym.alloc_chk;
              } else {
                alloc_func = lowersym.alloc;
              }
            }
          } else if ((XBIT(54, 0x1) || CLASSG(sptr) || is_or_has_poly(sptr) ||
                      has_finalized_component(sptr) || has_layout_desc(sptr)) &&
                     is_or_has_derived_allo(sptr)) {
            /* For -Mallocatable=03 where the object
             * is a derived type with an allocatable component
             * or with a nested or inherited allocatable component,
             * use calloc to make sure everything is zeroed out.
             * Otherwise, we may get UMRs on intrinsic assignments
             * from functions and structure constructors due to the
             * nested nature of the allocation.
             *
             * Also use calloc if object is derived type
             * declared CLASS, has a finalized component, or has a
             * polymorphic allocatable component.
             */
            if (lowersym.calloc == 0) {
                lowersym.calloc =
                    lower_makefunc(mkRteRtnNm(RTE_calloc04a), DT_NONE, FALSE);
            }
            alloc_func = lowersym.calloc;
          } else {
            if (lowersym.alloc == 0) {
                lowersym.alloc =
                    lower_makefunc(mkRteRtnNm(RTE_alloc04a), DT_NONE, FALSE);
            }
            if (lowersym.alloc_chk == 0) {
                lowersym.alloc_chk = lower_makefunc(
                    mkRteRtnNm(RTE_alloc04_chka), DT_NONE, FALSE);
            }
            if (ALLOCATTRG(sptr)) {
              alloc_func = lowersym.alloc_chk;
            } else {
              alloc_func = lowersym.alloc;
            }
          }
        }
      } else {
        /* Allocating POINTER arrays or automatic arrays:
         * RTE_ptr_alloc() does not check if the pointer is
         * already allocated
         */
        int src_ast;
        if (A_STARTG(ast)) {
          src_ast = A_STARTG(ast);
        try_next_src_ast:
          switch (A_TYPEG(src_ast)) {
          case A_ID:
          case A_LABEL:
          case A_ENTRY:
          case A_SUBSCR:
          case A_SUBSTR:
          case A_MEM:
            break;
          case A_CNST:
            src_ast = 0;
            break;
          case A_FUNC:
            src_ast = A_LOPG(src_ast);
            goto try_next_src_ast;
          default:
            interr("lower_stmt:unexp.ast", src_ast, 4);
          }
        } else {
          src_ast = 0;
        }
        src_dsc = 0;
        if (src_ast && (CLASSG(memsym_of_ast(src_ast)) || CLASSG(sptr))) {
          int src_sptr, src_dtype;

          src_sptr = memsym_of_ast(src_ast);
          src_dtype = DTYPEG(src_sptr);

          if (DTY(src_dtype) == TY_ARRAY)
            src_dtype = DTY(src_dtype + 1);
          if (DTY(src_dtype) == TY_DERIVED) {

            if (STYPEG(src_sptr) != ST_MEMBER) {
              if (!ALLOCDESCG(src_sptr) && !CLASSG(src_sptr)) {
                int dty = DTYPEG(src_sptr);
                src_dsc = get_static_type_descriptor(DTY(dty + 3));
              } else {
                src_dsc = get_type_descr_arg(gbl.currsub, src_sptr);
              }
            } else if (!CLASSG(src_sptr)) {
              int dty = DTYPEG(src_sptr);
              src_dsc = get_static_type_descriptor(DTY(dty + 3));
            } else {
              int sdsc_mem = SYMLKG(src_sptr);
              if (sdsc_mem == MIDNUMG(src_sptr) || PTRVG(sdsc_mem)) {
                sdsc_mem = SYMLKG(sdsc_mem);
                if (PTRVG(sdsc_mem) || !DESCARRAYG(sdsc_mem))
                  sdsc_mem = SYMLKG(sdsc_mem);
              }
              src_dsc = sdsc_mem;
              src_dsc_ast = mk_member(A_PARENTG(A_STARTG(ast)), mk_id(src_dsc),
                                      A_DTYPEG(A_PARENTG(A_STARTG(ast))));
              A_NDTYPEP(src_dsc_ast, A_DTYPEG(src_dsc_ast));
            }

            lowersym.ptr_alloc = 0;
              alloc_func = lower_makefunc(mkRteRtnNm(RTE_ptr_src_alloc04a),
                                          DT_NONE, FALSE);
          } else {
            if (lowersym.ptr_alloc == 0) {
                lowersym.ptr_alloc = lower_makefunc(
                    mkRteRtnNm(RTE_ptr_alloc04a), DT_NONE, FALSE);
            }
            alloc_func = lowersym.ptr_alloc;
          }

        } else {
          if (lowersym.ptr_alloc == 0) {
              lowersym.ptr_alloc =
                  lower_makefunc(mkRteRtnNm(RTE_ptr_alloc04a), DT_NONE, FALSE);
          }
          alloc_func = lowersym.ptr_alloc;
        }
        have_ptr_alloc = 1;
      }

      if (isarray && SDSCG(sptr) &&
          (ALIGNG(sptr) || DISTG(sptr) || POINTERG(sptr)) &&
          STYPEG(SDSCG(sptr)) != ST_PARAM) {
        eltype = DTY(dtype + 1);
        if (numilm == 0) {
          /* call lowersym.alloc with descriptor local size and stat argument */
          int sdsc, silm;
          sdsc = SDSCG(sptr);
          lower_visit_symbol(sdsc);
          numilm = lower_replacement(src, sdsc);
          silm = plower("oS", lowersym.bnd.con,
                        lower_getiszcon(get_lsize_index()));
          numilm = plower("onidi", "ELEMENT", 1, numilm, DTYPEG(sdsc), silm);
          if (size_of(lowersym.bnd.dtype) < 8) {
            numilm = lower_typeload(lowersym.bnd.dtype, numilm);
            numilm = plower("oi", "ITOI8", numilm);
          }
        }
      } else if (isarray && LNRZDG(sptr) && XBIT(52, 4)) {
        eltype = DTY(dtype + 1);
        /* generate the bounds from the subscripts */
        if (A_TYPEG(src) != A_SUBSCR) {
          ast_error("linearized allocate without bounds", ast);
          return;
        }
        if (numilm == 0) {
          /* generate ilm holding the size */
          asd = A_ASDG(src);
          ndim = ASD_NDIM(asd);
          for (i = 0; i < ndim; ++i) {
            int lb, ub, lbilm, ubilm;
            int ss = ASD_SUBS(asd, i);
            if (A_TYPEG(ss) == A_TRIPLE) {
              lb = A_LBDG(ss);
              ub = A_UPBDG(ss);
            } else {
              lb = 0;
              ub = ss;
            }
            lower_expression(ub);
            ubilm = lower_ilm(ub);
            if (lb && lb != astb.bnd.one) {
              if (lb != astb.bnd.zero) {
                lower_expression(lb);
                lbilm = lower_ilm(lb);
                ubilm = plower("oii", lowersym.bnd.sub, ubilm, lbilm);
              }
              lower_expression(astb.bnd.one);
              lbilm = lower_ilm(astb.bnd.one);
              ubilm = plower("oii", lowersym.bnd.add, ubilm, lbilm);
            }
            if (!numilm) {
              numilm = ubilm;
            } else {
              numilm = plower("oii", lowersym.bnd.mul, numilm, ubilm);
            }
          }
        }
      } else if (isarray) {
        eltype = DTY(dtype + 1);
        if (numilm == 0) {
          /* call lowersym.alloc with size and stat argument */
          DTYPE dty = dtype;
          int start = A_STARTG(ast);
          if (start) {
            if (is_array_dtype(A_DTYPEG(start))) {
              /* Avoid calling memsym_of_ast() on the result of an expression */
              dty = A_DTYPEG(start);
            } else {
              int src_sptr = memsym_of_ast(start);
              if (src_sptr > NOSYM && is_array_dtype(DTYPEG(src_sptr)))
                dty = DTYPEG(src_sptr);
            }
          }
          numilm = lower_numelm(dty, ast, src, 0);
          if (numilm == 0)
            return;
        }
      } else {
        numilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
        if (size_of(lowersym.bnd.dtype) < 8) {
          numilm = plower("oi", "ITOI8", numilm);
        }
        eltype = dtype;
        if (DTY(dtype) == TY_PTR) {
          eltype = DTY(eltype + 1);
        }
      }

      int src_ast = A_STARTG(ast);
      SPTR src_sptr = ast_is_sym(src_ast) ? memsym_of_ast(src_ast) : NOSYM;
      /* size of each element */
      if (DDTG(DTYPEG(sptr)) == DT_DEFERCHAR ||
          DDTG(DTYPEG(sptr)) == DT_DEFERNCHAR ||
          DDTG(A_DTYPEG(A_SRCG(ast))) == DT_DEFERCHAR ||
          DDTG(A_DTYPEG(A_SRCG(ast))) == DT_DEFERNCHAR) {
        if (LENG(sptr)) {
          size = mk_bnd_int(LENG(sptr));
        } else {
          size = string_expr_length(A_SRCG(ast));
        }
      } else if (src_sptr > NOSYM && CLASSG(sptr)) {
        size = mk_bnd_int(size_ast(src_sptr, eltype));
      } else {
        size = mk_bnd_int(size_ast(sptr, eltype));
      }
      lower_expression(size);
      sizeilm = lower_ilm(size);
      /* datatype */
      tyilm =
          plower("oS", lowersym.bnd.con, lower_getiszcon(dtype_to_arg(eltype)));
      /* stat */
      if (A_LOPG(ast)) {
        lower_expression(A_LOPG(ast));
        silm = lower_base(A_LOPG(ast));
      } else {
        silm = lower_null_arg();
      }
      /* pointer; check for scalar pointers */
      lower_expression(object);
      lower_disable_ptr_chk = 1;
      if (DTY(dtype) == TY_PTR) {
        /* scalar pointer, a la MEMBER */
        lilm = lower_target(object);
      } else {
        if (MIDNUMG(sptr) == 0) {
          lilm = lower_base(object);
        } else {
          lilm = lower_replacement(object, MIDNUMG(sptr));
        }
      }
      if (errmsg) {
        lower_expression(errmsg);
        errilm = lower_base(errmsg);
      } else
        errilm = lower_nullc_arg();
      if (A_FIRSTALLOCG(ast))
        firstilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
      else
        firstilm = plower("oS", lowersym.bnd.con, lowersym.bnd.zero);

      if (DTY(eltype) == TY_DERIVED) {
        int tag = DTY(eltype + 3);
        if (!XBIT(58, 0x20000000) && tag && POINTERG(tag)) {
          if (have_ptr_alloc) {
            if (lowersym.ptr_calloc == 0) {
                lowersym.ptr_calloc = lower_makefunc(
                    mkRteRtnNm(RTE_ptr_calloc04a), DT_NONE, FALSE);
            }
            alloc_func = lowersym.ptr_calloc;
          } else {
            if (lowersym.calloc == 0) {
                lowersym.calloc =
                    lower_makefunc(mkRteRtnNm(RTE_calloc04a), DT_NONE, FALSE);
            }
            alloc_func = lowersym.calloc;
          }
        }
      }

      lower_disable_ptr_chk = 0;
      if (lite_alloc && have_ptr_alloc && isarray && AUTOBJG(sptr)) {
        int ff;
        if (DTY(eltype) != TY_DERIVED || XBIT(58, 0x20000000) ||
            DTY(eltype + 3) == 0 || !POINTERG(DTY(eltype + 3))) {
          {
            if (lowersym.auto_alloc == 0) {
                lowersym.auto_alloc =
                    lower_makefunc(mkRteRtnNm(RTE_auto_alloc04), DT_PTR, FALSE);
            }
            ff = lowersym.auto_alloc;
            if (XBIT(54, 0x08)) {
              /*  place automatic arrays on the stack  */
              ff = 0;
            }
          }
        } else {
          if (lowersym.auto_calloc == 0) {
              lowersym.auto_calloc =
                  lower_makefunc(mkRteRtnNm(RTE_auto_calloc04), DT_PTR, FALSE);
          }
          ff = lowersym.auto_calloc;
        }
        if (!ff) {
          /*
           * ALLOCA's numilm is the same bitness as the target.
           * May need to adjust numilm as returned by lower_numelm():
           * +  if constant and the 64-bit target, numilm could be
           *    an ICON as opposed to KCON
           * +  if temp, numilm may be an address (BASE); need to load
           *    the temp.
           */
          if (numelm_constant) {
            if (lowersym.bnd.dtype != DT_INT8)
              numilm = plower("oi", "ITOI8", numilm);
          } else {
            if (XBIT(68, 0x1))
              /*
               * ugh, with -Mlarge_arrays, numilm is passed by
               * address to auto_alloc04_i8
               */
              numilm = lower_typeload(lowersym.bnd.dtype, numilm);
          }
          rilm = plower("oiisn", "ALLOCA", numilm, sizeilm, sptr, 0);
        } else {
          if (!XBIT(49, 0x20000000))
            rilm = plower("onsm", "PFUNC", 2, ff);
          else if (XBIT(49, 0x100)) /* 64-bit pointers */
            rilm = plower("onsm", "KFUNC", 2, ff);
          else
            rilm = plower("onsm", "IFUNC", 2, ff);
          plower("im", numilm);
          plower("im", sizeilm);
          plower("C", ff);
        }
        if (!XBIT(49, 0x20000000)) {
          ilm = plower("oii", "PST", lilm, rilm);
        } else if (XBIT(49, 0x100)) { /* 64-bit pointers */
          ilm = plower("oii", "KST", lilm, rilm);
        } else {
          ilm = plower("oii", "IST", lilm, rilm);
        }
      }
      else {
        if (!aln) {
          aln = astb.k0;
        }
        lower_expression(aln);
        alnilm = lower_ilm(aln);
        nullilm = lower_null_arg();
            if (src_dsc) {
          int src_ilm;
          if (!src_dsc_ast) {
            int src_sptr;
            if (A_STARTG(ast) && (A_TYPEG(A_STARTG(ast)) == A_ID ||
                                  A_TYPEG(A_STARTG(ast)) == A_MEM ||
                                  A_TYPEG(A_STARTG(ast)) == A_SUBSCR)) {
              src_sptr = memsym_of_ast(A_STARTG(ast));
              if (DSCASTG(src_sptr)) {
                src_ilm = lower_ilm(DSCASTG(src_sptr));
              } else
                src_ilm = plower("os", "BASE", src_dsc);
            } else
              src_ilm = plower("os", "BASE", src_dsc);
          } else {
            src_ilm = lower_ilm(src_dsc_ast);
            src_dsc_ast = 0;
          }
          plower("onsiiiiiiiiiiC", "CALL", 10, alloc_func, src_ilm, numilm,
                 tyilm, sizeilm, silm, lilm, nullilm, firstilm, alnilm, errilm,
                 alloc_func);
          src_dsc = 0;
        } else
          plower("onsiiiiiiiiiC", "CALL", 9, alloc_func, numilm, tyilm, sizeilm,
                 silm, lilm, nullilm, firstilm, alnilm, errilm, alloc_func);
      }
    } else { /* DEALLOCATE */
      int src, object, sptr, silm, ailm, lilm, symhandle, errmsg, errilm,
          firstilm, poly_dsc, dealloc_poly_func, poly_dsc_ast;
      poly_dsc = 0;
      poly_dsc_ast = 0;
      symhandle = 0;
      dealloc_poly_func = 0;
      src = A_SRCG(ast);
      if (A_TYPEG(src) == A_SUBSCR)
        object = A_LOPG(src);
      else
        object = src;
      sptr = find_pointer_variable(object);
      lower_visit_symbol(sptr);

      if (A_LOPG(ast)) {
        lower_expression(A_LOPG(ast));
        silm = lower_base(A_LOPG(ast));
      } else {
        silm = lower_null_arg();
      }
      errmsg = A_M3G(ast);
      if (errmsg) {
        lower_expression(errmsg);
        errilm = lower_base(errmsg);
      } else
        errilm = lower_nullc_arg();
      if (A_FIRSTALLOCG(ast))
        firstilm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
      else
        firstilm = plower("oS", lowersym.bnd.con, lowersym.bnd.zero);

      if (A_DALLOCMEMG(ast)) {
            if (lowersym.dealloc_mbr == 0 || is_or_has_poly(sptr) ||
                has_finalized_component(sptr)) {
          if (is_or_has_poly(sptr) || has_finalized_component(sptr)) {
            dealloc_poly_func = lower_makefunc(
                mkRteRtnNm(RTE_dealloc_poly_mbr03a), DT_NONE, FALSE);
            if (STYPEG(sptr) != ST_MEMBER) {
              poly_dsc = get_type_descr_arg(gbl.currsub, sptr);
            } else if (!CLASSG(sptr)) {
              int dty = DTYPEG(sptr);
              if (DTY(dty) == TY_ARRAY)
                dty = DTY(dty + 1);
              poly_dsc = get_static_type_descriptor(DTY(dty + 3));
            } else {
              int sdsc_mem = SYMLKG(sptr);
              if (sdsc_mem == MIDNUMG(sptr) || PTRVG(sdsc_mem)) {
                sdsc_mem = SYMLKG(sdsc_mem);
                if (PTRVG(sdsc_mem) || !DESCARRAYG(sdsc_mem))
                  sdsc_mem = SYMLKG(sdsc_mem);
              }
              poly_dsc = sdsc_mem;

              poly_dsc_ast = mk_member(A_PARENTG(A_SRCG(ast)), mk_id(poly_dsc),
                                       A_DTYPEG(A_PARENTG(A_SRCG(ast))));
              A_NDTYPEP(poly_dsc_ast, A_DTYPEG(poly_dsc_ast));
            }
          } else {
              lowersym.dealloc_mbr = lower_makefunc(
                  mkRteRtnNm(RTE_dealloc_mbr03a), DT_NONE, FALSE);
          }
        }
      } else {
        {
          if (lowersym.dealloc == 0 || is_or_has_poly(sptr) ||
              has_finalized_component(sptr)) {
            if (is_or_has_poly(sptr) || has_finalized_component(sptr)) {
              lowersym.dealloc = lower_makefunc(mkRteRtnNm(RTE_dealloc_poly03),
                                                DT_NONE, FALSE);
              if (DTY(DTYPEG(sptr)) == TY_ARRAY) {
                poly_dsc = (STYPEG(sptr) != ST_MEMBER)
                               ? get_type_descr_arg(gbl.currsub, sptr)
                               : SDSCG(sptr);
                if (STYPEG(poly_dsc) == ST_MEMBER) {
                  poly_dsc_ast = mk_member(A_PARENTG(src), mk_id(poly_dsc),
                                           A_DTYPEG(A_PARENTG(src)));
                  A_NDTYPEP(poly_dsc_ast, A_DTYPEG(poly_dsc_ast));
                }
              } else if (CLASSG(sptr)) {
                if (STYPEG(sptr) != ST_MEMBER) {
                  poly_dsc = get_type_descr_arg(gbl.currsub, sptr);
                } else {
                  /* member case */
                  int sdsc_mem = SYMLKG(sptr);
                  if (sdsc_mem == MIDNUMG(sptr) || PTRVG(sdsc_mem)) {
                    sdsc_mem = SYMLKG(sdsc_mem);
                    if (PTRVG(sdsc_mem) || !DESCARRAYG(sdsc_mem))
                      sdsc_mem = SYMLKG(sdsc_mem);
                  }
                  poly_dsc = sdsc_mem;
                  poly_dsc_ast =
                      mk_member(A_PARENTG(A_SRCG(ast)), mk_id(poly_dsc),
                                A_DTYPEG(A_PARENTG(A_SRCG(ast))));
                  A_NDTYPEP(poly_dsc_ast, A_DTYPEG(poly_dsc_ast));
                }
              } else {
                int dty = DTYPEG(sptr);
                if (DTY(dty) == TY_ARRAY)
                  dty = DTY(dty + 1);
                if (DTY(dty) == TY_DERIVED) {
                  int tag = DTY(dty + 3);
                  poly_dsc = get_static_type_descriptor(tag);
                } else
                  poly_dsc = get_static_type_descriptor(sptr);
              }
            } else
              lowersym.dealloc =
                  lower_makefunc(mkRteRtnNm(RTE_dealloc03a), DT_NONE, FALSE);
          }
          dealloc_func = lowersym.dealloc;
          if (poly_dsc)
            lowersym.dealloc = 0;
        }
      }
      dtype = DTYPEG(sptr);
      lower_expression(object);
      lower_disable_ptr_chk = 1;
      lilm = lower_base_address(object, ArgumentBase);
      lower_disable_ptr_chk = 0;
      if (A_DALLOCMEMG(ast)) {
        if (poly_dsc) {
          int poly_ilm;
          if (!poly_dsc_ast) {
            poly_ilm = plower("os", "BASE", poly_dsc);
          } else {
            poly_ilm = lower_ilm(poly_dsc_ast);
            poly_dsc_ast = 0;
          }
          ailm = plower("onsiiiiC", "CALL", 4, dealloc_poly_func, poly_ilm,
                        silm, lilm, errilm, lowersym.dealloc_mbr);
        } else {
          ailm = plower("onsiiiC", "CALL", 3, lowersym.dealloc_mbr, silm, lilm,
                        errilm, lowersym.dealloc_mbr);
        }
      } else if (lite_alloc && AUTOBJG(sptr) && DTY(DTYPEG(sptr)) == TY_ARRAY) {
        int ff;
        {
          if (lowersym.auto_dealloc == 0) {
              lowersym.auto_dealloc =
                  lower_makefunc(mkRteRtnNm(RTE_auto_dealloc), DT_NONE, FALSE);
          }
          ff = lowersym.auto_dealloc;
          if (XBIT(54, 0x08)) {
            ailm = plower("oissn", "DEALLOCA", lilm, sptr, ff, 0);
            ff = 0;
          }
        }
        if (ff)
          ailm = plower("onsiC", "CALL", 1, ff, lilm, ff);
      } else {
            if (poly_dsc) {
          int poly_ilm;
          if (!poly_dsc_ast) {
            poly_ilm = plower("os", "BASE", poly_dsc);
          } else {
            poly_ilm = lower_ilm(poly_dsc_ast);
            poly_dsc_ast = 0;
          }
          ailm = plower("onsiiiiiC", "CALL", 5, dealloc_func, poly_ilm, silm,
                        lilm, firstilm, errilm, dealloc_func);
        } else {
          ailm = plower("onsiiiiC", "CALL", 4, dealloc_func, silm, lilm,
                        firstilm, errilm, dealloc_func);
        }
      }
      lower_reinit();
      add_nullify(object);
    }
    lower_end_stmt(std);
    break;

  case A_ASN:
    lower_start_stmt(lineno, label, TRUE, std);
    dest = A_DESTG(ast);
    lower_expression(dest);
    lilm = lower_base(dest);
    lower_reinit();
    src = A_SRCG(ast);
    if (A_TYPEG(src) == A_UNOP && A_OPTYPEG(src) == OP_LOC) {
      /* disable pointer check if this is a
       * pointer assign, not a dereference.
       */
      int saved_val = lower_disable_ptr_chk;
      lower_disable_ptr_chk = 1;
      lower_expression(src);
      lower_disable_ptr_chk = saved_val;
    } else {
      lower_expression(src);
    }
    /* sometimes the assignment itself has no type, probably a bug */
    dtype = A_NDTYPEG(dest);

    if (DTY(dtype) == TY_DERIVED || DTY(dtype) == TY_STRUCT) {
      /* sometimes the compiler allows an INT to be assigned
       * to a simple derived type */
      dtype = A_NDTYPEG(src);

      if ((DTY(dtype) == TY_DERIVED || DTY(dtype) == TY_STRUCT)) {
        int dtype2 = A_DTYPEG(ast);
        if (DTY(dtype2) == TY_DERIVED || DTY(dtype2) == TY_STRUCT) {
          dtype = dtype2;
        }
      }
    }
    if (DTYG(dtype) == TY_STRUCT || DTYG(dtype) == TY_DERIVED) {
      rilm = lower_base(src);
    } else if (dtype) {
      rilm = lower_conv(src, dtype);
    } else {
      rilm = lower_ilm(src);
    }
    if (!XBIT(49, 0x20000000)) {
      switch (A_TYPEG(dest)) {
      case A_ID:
        sptr = A_SPTRG(dest);
        if (PTRVG(sptr) && HCCSYMG(sptr)) {
          dtype = DT_ADDR;
        }
        break;
      case A_MEM:
        if (PTRVG(A_SPTRG(A_MEMG(dest)))) {
          dtype = DT_ADDR;
        }
        break;
      }
    }

    switch (DTYG(dtype)) {
    case TY_BINT:
    case TY_BLOG:
      plower("oii", "CHST", lilm, rilm);
      break;
    case TY_SINT:
      plower("oii", "SIST", lilm, rilm);
      break;
    case TY_INT:
      plower("oii", "IST", lilm, rilm);
      break;
    case TY_PTR:
      if (dtype != DT_ADDR || !XBIT(49, 0x20000000)) {
        /* real pointer type */
        plower("oii", "PST", lilm, rilm);
      } else if (XBIT(49, 0x100)) { /* 64-bit pointers */
        plower("oii", "KST", lilm, rilm);
      } else {
        plower("oii", "IST", lilm, rilm);
      }
      break;
    case TY_INT8:
      plower("oii", "KST", lilm, rilm);
      break;
    case TY_SLOG:
      plower("oii", "SLST", lilm, rilm);
      break;
    case TY_LOG:
      plower("oii", "LST", lilm, rilm);
      break;
    case TY_LOG8:
      plower("oii", "KLST", lilm, rilm);
      break;
    case TY_REAL:
      plower("oii", "RST", lilm, rilm);
      break;
    case TY_DBLE:
      plower("oii", "DST", lilm, rilm);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    /* to support quad precision store in ilm */
    case TY_QUAD:
      plower("oii", "QFST", lilm, rilm);
      break;
#endif
    case TY_CMPLX:
      plower("oii", "CST", lilm, rilm);
      break;
    case TY_DCMPLX:
      plower("oii", "CDST", lilm, rilm);
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QCMPLX:
      plower("oii", "CQST", lilm, rilm);
      break;
#endif
    case TY_CHAR:
      plower("oii", "SST", lilm, rilm);
      break;
    case TY_NCHAR:
      plower("oii", "NSST", lilm, rilm);
      break;
    case TY_DERIVED:
    case TY_STRUCT:
      plower("oiid", "SMOVE", lilm, rilm, dtype);
      break;
    default:
      ast_error("unexpected data type at assignment", ast);
      break;
    }
    lower_end_stmt(std);
    break;

  case A_ASNGOTO:
    /* can be either assign of GOTO label or FORMAT label */
    rop = A_SRCG(ast);
    lop = A_DESTG(ast);
    fmt = FMTPTG(A_SPTRG(rop));
    lower_start_stmt(lineno, label, TRUE, std);
    if (fmt) {
      /* format assign */
      lower_expression(lop);
      lilm = lower_base(lop);
      rilm = plower("oS", "BASE", fmt);
    } else {
/* GOTO assign */
      assert(AGOTOG(A_SPTRG(rop)), "lower_stmt:A_ASNGOTO AGOTO value 0",
             A_SPTRG(rop), 3);
      rilm = plower("os", "ICON", lower_getintcon(AGOTOG(A_SPTRG(rop))));
      if (A_DTYPEG(lop) == DT_INT8) {
        rilm = plower("oi", "ITOI8", rilm);
      }
      lower_expression(lop);
      lilm = lower_base(lop);
      lower_typestore(A_DTYPEG(lop), lilm, rilm);
      lower_end_stmt(std);
      break;
    }
    if (XBIT(49, 0x100))
      /* 64-bit pointers */
      plower("oii", "KAST", lilm, rilm);
    else
      plower("oii", "AST", lilm, rilm);
    lower_end_stmt(std);
    break;

  case A_ICALL:
    switch (A_OPTYPEG(ast)) {
    case I_MVBITS:
      lower_start_stmt(lineno, label, TRUE, std);
      count = A_ARGCNTG(ast);
      args = A_ARGSG(ast);
      symfunc = lower_makefunc(mkRteRtnNm(RTE_mvbits), DT_NONE, FALSE);
      num = 0;
      for (i = 0; i < 5; ++i) {
        lower_expression(ARGT_ARG(args, i));
        symargs[num++] = lower_base(ARGT_ARG(args, i));
      }
      lop = ARGT_ARG(args, 0);
      dtype = A_NDTYPEG(lop);
      symargs[num++] =
          plower("os", "ICON", lower_getintcon(size_of(DDTG(dtype))));
      lop = ARGT_ARG(args, 1);
      dtype = A_NDTYPEG(lop);
      symargs[num++] =
          plower("os", "ICON", lower_getintcon(size_of(DDTG(dtype))));
      lop = ARGT_ARG(args, 2);
      dtype = A_NDTYPEG(lop);
      symargs[num++] =
          plower("os", "ICON", lower_getintcon(size_of(DDTG(dtype))));
      lop = ARGT_ARG(args, 4);
      dtype = A_NDTYPEG(lop);
      symargs[num++] =
          plower("os", "ICON", lower_getintcon(size_of(DDTG(dtype))));
      ilm = plower("onsm", "CALL", num, symfunc);
      for (i = 0; i < num; ++i) {
        plower("im", symargs[i]);
      }
      plower("C", symfunc);
      lower_end_stmt(std);
      return;
    case I_PTR2_ASSIGN:
      /* generate assignment for scalar pointer inline;
       * call ptr_assign for array pointers */
      is_assumeshp = 0;
      count = A_ARGCNTG(ast);
      args = A_ARGSG(ast);
      lop2 = lop = ARGT_ARG(args, 0);
      rop = ARGT_ARG(args, count > 2 ? 2 : 1);
    again:
      if (A_TYPEG(lop) == A_ID) {
        sym = A_SPTRG(lop);
      } else if (A_TYPEG(lop) == A_MEM) {
        sym = A_SPTRG(A_MEMG(lop));
      } else if (A_TYPEG(lop) == A_SUBSCR) {
        lop = A_LOPG(lop);
        goto again;
      } else {
        lerror("unsupported pointer assignment target");
        return;
      }
      if (A_TYPEG(rop) == A_INTR && A_OPTYPEG(rop) == I_NULL) {
        /* <ptr>$p = 0 */
        lower_start_stmt(lineno, label, TRUE, std);
        lower_disable_ptr_chk = 1;
        lower_expression(lop);
        lilm = lower_target(lop);
        lower_disable_ptr_chk = 0;
        lower_reinit();
        rilm = lower_null();
        lower_typestore(DT_ADDR, lilm, rilm);
        lower_reinit();
        lower_end_stmt(std);
      } else if (DTY(DTYPEG(sym)) != TY_ARRAY) {
        int lopsdsc, ropsdsc, dtype;
        lopsdsc = ARGT_ARG(args, 1);
        ropsdsc = ARGT_ARG(args, 3);
        lower_start_stmt(lineno, label, TRUE, std);
        lower_disable_ptr_chk = 1;
        lower_expression(lop);
        lilm = lower_target(lop);
        lower_disable_ptr_chk = 0;
        lower_reinit();
        lower_expression(rop);
        rilm = lower_address(rop);
        lower_typestore(DT_ADDR, lilm, rilm);
        doit = 0;
        if (A_TYPEG(lopsdsc) == A_ID) {
          int lopsptr = A_SPTRG(lopsdsc);
          if (STYPEG(lopsptr) != ST_PARAM &&
              (STYPEG(sym) != ST_MEMBER || STYPEG(lopsptr) == ST_MEMBER)) {
            /* don't if sdsc is PARAM or if sym is member and
             * sdsc is not a member */
            doit = 1;
          }
        } else if (A_TYPEG(lopsdsc) == A_MEM || A_TYPEG(lopsdsc) == A_SUBSCR) {
          doit = 1;
        }
        if (doit) {
          lower_expression(lopsdsc);
          lilm = lower_base(lopsdsc);
          dtype = A_NDTYPEG(lopsdsc);
          if (A_TYPEG(lopsdsc) == A_ID &&
              DTY(DTYPEG(A_SPTRG(lopsdsc))) == TY_ARRAY) {
            /* get first element */
            silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
            dtype = DTYPEG(A_SPTRG(lopsdsc));
            lilm = plower("onidi", "ELEMENT", 1, lilm, dtype, silm);
            dtype = DDTG(dtype);
          }
          lower_reinit();
          lower_expression(ropsdsc);
          if (A_TYPEG(ropsdsc) == A_ID &&
              DTY(DTYPEG(A_SPTRG(ropsdsc))) == TY_ARRAY) {
            int dtype;
            /* get first element */
            rilm = lower_base(ropsdsc);
            silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
            dtype = DTYPEG(A_SPTRG(ropsdsc));
            rilm = plower("onidi", "ELEMENT", 1, rilm, dtype, silm);
            rilm = lower_typeload(dtype, rilm);
          } else {
            rilm = lower_ilm(ropsdsc);
          }
          lower_typestore(dtype, lilm, rilm);
        }
        lower_end_stmt(std);
      } else if (bnds_remap_list(lop2) && A_TYPEG(rop) == A_SUBSCR) {
        /* rop w/subscr ==> ptr reshape asn */
        lower_start_stmt(lineno, label, TRUE, std);
        lower_expression(lop);
        lilm = lower_target(lop);
        lower_reinit();
        rop = first_element(rop);
        lower_expression(rop);
        rilm = lower_address(rop);
        lower_typestore(DT_ADDR, lilm, rilm);
        lower_end_stmt(std);
      } else {
        int ropsym = memsym_of_ast(rop);
        int ptrsym, dscsym;

        if (A_TYPEG(lop2) == A_SUBSCR) { /* ptr reshape */
          int bounds, rank;
          rank = rank_of_sym(sym);
          bounds = (count - 8);
          if (bounds == rank) {
            symfunc =
                lower_makefunc(mkRteRtnNm(RTE_ptr_shape_assnx), DT_PTR, FALSE);
          } else {
            symfunc =
                lower_makefunc(mkRteRtnNm(RTE_ptr_shape_assn), DT_PTR, FALSE);
          }
        } else
          switch (DTYG(DTYPEG(sym))) {
          case TY_CHAR:
          case TY_NCHAR:
            if (ASSUMSHPG(ropsym)) {
              if (DDTG(DTYPEG(sym)) == DT_DEFERCHAR ||
                  DDTG(DTYPEG(sym)) == DT_DEFERNCHAR) {
                symfunc = lower_makefunc(
                    mkRteRtnNm(RTE_ptr_assn_dchar_assumeshp), DT_PTR, FALSE);
                is_assumeshp = 1;
              } else {
                symfunc = lower_makefunc(
                    mkRteRtnNm(RTE_ptr_assn_char_assumeshp), DT_PTR, FALSE);
                is_assumeshp = 1;
              }
            } else {
              if (DDTG(DTYPEG(sym)) == DT_DEFERCHAR ||
                  DDTG(DTYPEG(sym)) == DT_DEFERNCHAR) {
                symfunc = lower_makefunc(count == 5
                                             ? mkRteRtnNm(RTE_ptr_assn_dchara)
                                             : mkRteRtnNm(RTE_ptr_assn_dcharxa),
                                         DT_PTR, FALSE);
              } else {
                symfunc =
                    lower_makefunc(count == 5 ? mkRteRtnNm(RTE_ptr_assn_chara)
                                              : mkRteRtnNm(RTE_ptr_assn_charxa),
                                   DT_PTR, FALSE);
              }
            }
            break;
          default:
            if (ASSUMSHPG(ropsym) &&
                (!XBIT(58, 0x400000) || !TARGETG(ropsym))) {
              symfunc = lower_makefunc(mkRteRtnNm(RTE_ptr_assn_assumeshp),
                                       DT_PTR, FALSE);
              is_assumeshp = 2;
            } else {
              symfunc = lower_makefunc(count == 5 ? mkRteRtnNm(RTE_ptr_assn)
                                                  : mkRteRtnNm(RTE_ptr_assnx),
                                       DT_PTR, FALSE);
            }
          }
        lower_start_stmt(lineno, label, TRUE, std);
        num = 0;
        lower_disable_ptr_chk = 1;
        lower_expression(lop);
        ptrsym = sym;
        symargs[num++] = lower_base(lop);
        lilm = lower_target(lop);
        lower_disable_ptr_chk = 0;
        lop = ARGT_ARG(args, 1);
        if (A_TYPEG(lop) == A_ID) {
          sym = A_SPTRG(lop);
        } else if (A_TYPEG(lop) == A_MEM) {
          sym = A_SPTRG(A_MEMG(lop));
        } else {
          lerror("unsupported array pointer assignment target");
          return;
        }
        lower_expression(lop);
        dscsym = sym;
        symargs[num++] = lower_base(lop);
        lop = ARGT_ARG(args, 2);
        if (STYPEG(sym) != ST_VAR && A_TYPEG(lop) == A_SUBSCR &&
            A_SHAPEG(lop)) {
          int llop;
          llop = A_LOPG(lop);
          if (A_TYPEG(llop) == A_ID ||
              (A_TYPEG(llop) == A_MEM && A_SHAPEG(A_PARENTG(llop)) == 0)) {
            lop = A_LOPG(lop);
          }
        }
        lower_disable_ptr_chk = 1;
        {
          lower_expression(lop);
          symargs[num++] = lower_base(lop);
        }
        lower_disable_ptr_chk = 0;
        rop = ARGT_ARG(args, 3);
        lower_expression(rop);
        symargs[num++] = lower_base(rop);
        for (i = 4; i < count; ++i) {
          lop = ARGT_ARG(args, i);
          lower_expression(lop);
          symargs[num++] = lower_base(lop);
        }
        if (!XBIT(49, 0x20000000))
          rilm = plower("onsm", "PFUNC", num, symfunc);
        else if (XBIT(49, 0x100)) /* 64-bit pointers */
          rilm = plower("onsm", "KFUNC", num, symfunc);
        else
          rilm = plower("onsm", "IFUNC", num, symfunc);
        for (i = 0; i < num; ++i) {
          plower("im", symargs[i]);
        }
        plower("C", symfunc);
        lower_typestore(DT_ADDR, lilm, rilm);
        if (is_assumeshp) {
          /* check section flag */
          lop = ARGT_ARG(args, count - 1);
          if (lop == astb.bnd.zero) {
            /* not a section => whole array */
            int base;
            dtype = DTYPEG(ropsym);
            ndim = ADD_NUMDIM(dtype);
            switch (ndim) {
            case 1:
              rtlRtn = RTE_ptr_fix_assumeshp1;
              break;
            case 2:
              rtlRtn = RTE_ptr_fix_assumeshp2;
              break;
            case 3:
              rtlRtn = RTE_ptr_fix_assumeshp3;
              break;
            default:
              rtlRtn = RTE_ptr_fix_assumeshp;
              break;
            }
            symargs[0] = lower_base(ARGT_ARG(args, 1));
            symfunc = lower_makefunc(mkRteRtnNm(rtlRtn), DT_NONE, FALSE);
            num = 1;
            if (ndim > 3) {
              ilm = plower("oS", lowersym.bnd.con, lower_getiszcon(ndim));
              symargs[num++] = plower("oi", "DPVAL", ilm);
            }
            for (i = 0; i < ndim; i++) {
              lop = ADD_LWAST(dtype, i);
              if (A_ALIASG(lop)) {
                sptr = A_ALIASG(lop);
                sptr = A_SPTRG(sptr);
                ilm = plower("oS", lowersym.bnd.con, sptr);
              } else {
                lower_expression(lop);
                base = lower_base(lop);
                ilm = lower_typeload(A_NDTYPEG(lop), base);
              }
              symargs[num++] = plower("oi", "DPVAL", ilm);
            }
            ilm = plower("onsm", "CALL", num, symfunc);
            for (i = 0; i < num; ++i) {
              plower("im", symargs[i]);
            }
            plower("C", symfunc);
          }
        }
        lower_end_stmt(std);
      }
      return;
    case I_PTR_COPYIN:
      count = A_ARGCNTG(ast);
      args = A_ARGSG(ast);
      lop = ARGT_ARG(args, 3);
      if (A_TYPEG(lop) == A_ID) {
        sym = A_SPTRG(lop);
      } else if (A_TYPEG(lop) == A_MEM) {
        sym = A_SPTRG(A_MEMG(lop));
      } else {
        lerror("internal error at subprogram interface; bad copy-in target");
        return;
      }
      if (DTY(DTYPEG(sym)) != TY_ARRAY) {
        /* do copyin in-line for scalar pointers */
        int lopsdsc, rop, ropsdsc, dtype;
        rop = ARGT_ARG(args, 5);
        lopsdsc = ARGT_ARG(args, 4);
        ropsdsc = ARGT_ARG(args, 6);
        lower_start_stmt(lineno, label, TRUE, std);
        lower_expression(lop);
        lilm = lower_target(lop);
        lower_reinit();
        lower_expression(rop);
        rilm = lower_target(rop);
        rilm = lower_typeload(DT_ADDR, rilm);
        lower_typestore(DT_ADDR, lilm, rilm);
        lower_expression(lopsdsc);
        lilm = lower_base(lopsdsc);
        dtype = A_NDTYPEG(lopsdsc);
        if (A_TYPEG(lopsdsc) == A_ID &&
            DTY(DTYPEG(A_SPTRG(lopsdsc))) == TY_ARRAY) {
          /* get first element */
          silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
          dtype = DTYPEG(A_SPTRG(lopsdsc));
          lilm = plower("onidi", "ELEMENT", 1, lilm, dtype, silm);
          dtype = DDTG(dtype);
        }
        lower_reinit();
        lower_expression(ropsdsc);
        if (A_TYPEG(ropsdsc) == A_ID &&
            DTY(DTYPEG(A_SPTRG(ropsdsc))) == TY_ARRAY) {
          /* get first element */
          int rdtype;
          rdtype = DTYPEG(A_SPTRG(ropsdsc));
          rilm = lower_base(ropsdsc);
          silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
          rilm = plower("onidi", "ELEMENT", 1, rilm, rdtype, silm);
          rilm = lower_typeload(DDTG(rdtype), rilm);
        } else {
          rilm = lower_ilm(ropsdsc);
        }
        lower_typestore(dtype, lilm, rilm);
        lower_end_stmt(std);
      } else {
        /* call ptr_in for array pointers */
        lower_start_stmt(lineno, label, TRUE, std);
        switch (DTYG(DTYPEG(sym))) {
        case TY_CHAR:
        case TY_NCHAR:
          symfunc =
              lower_makefunc(mkRteRtnNm(RTE_ptr_in_chara), DT_NONE, FALSE);
          break;
        default:
          symfunc = lower_makefunc(mkRteRtnNm(RTE_ptr_ina), DT_NONE, FALSE);
        }
        num = 0;
        for (i = 0; i < count && i < 7; ++i) {
          lower_expression(ARGT_ARG(args, i));
          if (i == 3) {
            lower_disable_ptr_chk = 1;
          }
          symargs[num++] = lower_base(ARGT_ARG(args, i));
          if (i == 3) {
            lower_disable_ptr_chk = 0;
          }
        }
        ilm = plower("onsm", "CALL", num, symfunc);
        for (i = 0; i < num; ++i) {
          plower("im", symargs[i]);
        }
        plower("C", symfunc);
        lower_end_stmt(std);
      }
      return;

    case I_PTR_COPYOUT:
      count = A_ARGCNTG(ast);
      args = A_ARGSG(ast);
      lop = ARGT_ARG(args, 2);
      if (A_TYPEG(lop) == A_ID) {
        sym = A_SPTRG(lop);
      } else if (A_TYPEG(lop) == A_MEM) {
        sym = A_SPTRG(A_MEMG(lop));
      } else {
        lerror("internal error at subprogram interface; bad copy-out target");
        return;
      }
      if (DTY(DTYPEG(sym)) != TY_ARRAY) {
        /* same code as for copy-in, but lop/rop reversed */
        int lopsdsc, rop, ropsdsc, dtype;
        rop = ARGT_ARG(args, 0);
        lopsdsc = ARGT_ARG(args, 3);
        ropsdsc = ARGT_ARG(args, 1);
        lower_start_stmt(lineno, label, TRUE, std);
        lower_expression(rop);
        lilm = lower_target(rop);
        lower_reinit();
        lower_expression(lop);
        rilm = lower_target(lop);
        rilm = lower_typeload(DT_ADDR, rilm);
        lower_typestore(DT_ADDR, lilm, rilm);
        lower_expression(ropsdsc);
        lilm = lower_base(ropsdsc);
        dtype = A_NDTYPEG(ropsdsc);
        if (A_TYPEG(ropsdsc) == A_ID &&
            DTY(DTYPEG(A_SPTRG(ropsdsc))) == TY_ARRAY) {
          /* get first element */
          silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
          dtype = DTYPEG(A_SPTRG(ropsdsc));
          lilm = plower("onidi", "ELEMENT", 1, lilm, dtype, silm);
          dtype = DDTG(dtype);
        }
        lower_reinit();
        lower_expression(lopsdsc);
        if (A_TYPEG(lopsdsc) == A_ID &&
            DTY(DTYPEG(A_SPTRG(lopsdsc))) == TY_ARRAY) {
          /* get first element */
          int rdtype;
          rilm = lower_base(lopsdsc);
          silm = plower("oS", lowersym.bnd.con, lowersym.bnd.one);
          rdtype = DTYPEG(A_SPTRG(lopsdsc));
          rilm = plower("onidi", "ELEMENT", 1, rilm, rdtype, silm);
          rilm = lower_typeload(DDTG(rdtype), rilm);
        } else {
          rilm = lower_ilm(lopsdsc);
        }
        lower_typestore(dtype, lilm, rilm);
        lower_end_stmt(std);
      } else {
        lower_start_stmt(lineno, label, TRUE, std);
        switch (DTYG(DTYPEG(sym))) {
        case TY_CHAR:
        case TY_NCHAR:
          symfunc =
              lower_makefunc(mkRteRtnNm(RTE_ptr_out_chara), DT_NONE, FALSE);
          break;
        default:
          symfunc = lower_makefunc(mkRteRtnNm(RTE_ptr_out), DT_NONE, FALSE);
        }
        num = 0;
        for (i = 0; i < count && i < 7; ++i) {
          lower_expression(ARGT_ARG(args, i));
          if (i == 2) {
            lower_disable_ptr_chk = 1;
          }
          symargs[num++] = lower_base(ARGT_ARG(args, i));
          if (i == 2) {
            lower_disable_ptr_chk = 0;
          }
        }
        ilm = plower("onsm", "CALL", num, symfunc);
        for (i = 0; i < num; ++i) {
          plower("im", symargs[i]);
        }
        plower("C", symfunc);
        lower_end_stmt(std);
      }
      return;

    case I_NULLIFY:
      args = A_ARGSG(ast);
      lop = ARGT_ARG(args, 0);
      lower_start_stmt(lineno, label, TRUE, std);
      add_nullify(lop);
      lower_end_stmt(std);
      return;
    case I_COPYIN:
      symfunc = lower_makefunc(mkRteRtnNm(RTE_qopy_in), DT_NONE, FALSE);
      lower_start_stmt(lineno, label, TRUE, std);
      handle_arguments(ast, symfunc, 0);
      lower_end_stmt(std);
      return;
    case I_COPYOUT:
      symfunc = lower_makefunc(mkRteRtnNm(RTE_copy_out), DT_NONE, FALSE);
      lower_start_stmt(lineno, label, TRUE, std);
      handle_arguments(ast, symfunc, 0);
      lower_end_stmt(std);
      return;
    default:
      /* default is to treat like a call statement,
       * fall through */
      break;
    }
    FLANG_FALLTHROUGH;
  case A_CALL:
    lower_start_stmt(lineno, label, TRUE, std);
    lop = A_LOPG(ast);
    symfunc = procsym_of_ast(lop);
    if (STYPEG(symfunc) == ST_MEMBER && CLASSG(symfunc) && CCSYMG(symfunc) &&
        VTABLEG(symfunc)) {
      symfunc = VTABLEG(symfunc);
    }
    if (!is_procedure_ptr(symfunc)) {
      if (HCCSYMG(symfunc)) {
        char *nm;
        nm = SYMNAME(symfunc);
        /*
         * Would be nice if there were a flag in the symbol table
         * to say inhibit subscript checking when calling certain
         * functions.
         */
        if (strcmp(nm, mkRteRtnNm(RTE_copy_f77_argl)) == 0 ||
            strcmp(nm, mkRteRtnNm(RTE_copy_f77_argsl)) == 0 ||
            strcmp(nm, mkRteRtnNm(RTE_copy_f90_argl)) == 0) {
          lower_disable_subscr_chk = 1;
        }
      }
      handle_arguments(ast, symfunc, 0);
    } else
      handle_arguments(ast, symfunc, 1);
    lower_end_stmt(std);
    lower_disable_subscr_chk = 0;
    break;

  case A_CGOTO:
    lab = lower_lab();
    count = 0;
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli))
      ++count;
    labnum = plabel(lab, count, 1);
    count = 0;
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      ++count;
      plabel(A_SPTRG(ASTLI_AST(astli)), count, 0);
    }
    lower_start_stmt(lineno, label, TRUE, std);
    lower_expression(A_LOPG(ast));
    ilm = lower_conv(A_LOPG(ast), DT_INT4);
    plower("oin", "CGOTO", ilm, labnum);
    plower("oL", "LABEL", lab);
    lower_end_stmt(std);
    break;

  case A_CONTINUE:
    /* no code generated for a CONTINUE statement */
    if (label) {
      lower_start_stmt(lineno, label, TRUE, std);
      lower_end_stmt(std);
    }
    break;

  case A_ENTRY:
    lab = lower_lab();
    lower_start_stmt(lineno, 0, TRUE, std);
    plower("oS", "BR", lab);
    plower("oS", "ENTRY", A_SPTRG(ast));
    lower_end_stmt(0);
    if (XBIT(52, 4) || XBIT(58, 0x22)) {
      fill_entry_bounds(A_SPTRG(ast), lineno);
    }
    lower_debug_label();
    lower_start_stmt(lineno, label, TRUE, 0);
    plower("oL", "LABEL", lab);
    lower_pointer_init();
    lower_end_stmt(std);
    break;

  case A_AIF:
    lower_start_stmt(lineno, 0, TRUE, std);
    lower_expression(A_IFEXPRG(ast));
    ilm = lower_ilm(A_IFEXPRG(ast));
    dtype = A_NDTYPEG(A_IFEXPRG(ast));
    plower("oisss", ltyped("AIF", dtype), ilm, A_SPTRG(A_L1G(ast)),
           A_SPTRG(A_L2G(ast)), A_SPTRG(A_L3G(ast)));
    lower_end_stmt(std);
    break;

  case A_IF:
    lower_start_stmt(lineno, label, TRUE, std);
    iflab.thenlabel = 0;
    iflab.elselabel = lower_lab();
    iflab.endlabel = iflab.elselabel;
    lower_logical(A_IFEXPRG(ast), &iflab);
    if (iflab.thenlabel) {
      plower("oL", "LABEL", iflab.thenlabel);
    }
    lower_end_stmt(0);
    lower_stmt(std, A_IFSTMTG(ast), lineno, 0);
    lower_start_stmt(lineno, 0, TRUE, 0);
    plower("oL", "LABEL", iflab.elselabel);
    lower_end_stmt(std);
    break;

  case A_IFTHEN:
    lower_start_stmt(lineno, label, TRUE, std);
    iflab.thenlabel = 0;
    iflab.elselabel = lower_lab();
    iflab.endlabel = iflab.elselabel;
    lower_logical(A_IFEXPRG(ast), &iflab);
    if (iflab.thenlabel) {
      plower("oL", "LABEL", iflab.thenlabel);
    }
    lower_push(iflab.elselabel);
    lower_push(iflab.endlabel);
    lower_push(STKIF);
    lower_end_stmt(std);
    break;

  case A_ELSE:
    lower_start_stmt(lineno, 0, TRUE, std);
    lower_check_stack(STKIF);
    iflab.endlabel = lower_pop();
    iflab.elselabel = lower_pop();
    if (iflab.endlabel == iflab.elselabel)
      iflab.endlabel = lower_lab();
    plower("oS", "BR", iflab.endlabel);
    lower_end_stmt(0);
    lower_start_stmt(lineno, label, TRUE, 0);
    plower("oL", "LABEL", iflab.elselabel);
    lower_push(iflab.endlabel);
    lower_push(iflab.endlabel);
    lower_push(STKIF);
    lower_end_stmt(std);
    break;

  case A_ELSEIF:
    lower_start_stmt(lineno, 0, TRUE, std);
    lower_check_stack(STKIF);
    iflab.endlabel = lower_pop();
    iflab.elselabel = lower_pop();
    if (iflab.endlabel == iflab.elselabel)
      iflab.endlabel = lower_lab();
    plower("oS", "BR", iflab.endlabel);
    lower_end_stmt(0);
    lower_start_stmt(lineno, label, TRUE, 0);
    plower("oL", "LABEL", iflab.elselabel);
    iflab.thenlabel = 0;
    iflab.elselabel = lower_lab();
    lower_logical(A_IFEXPRG(ast), &iflab);
    if (iflab.thenlabel) {
      plower("oL", "LABEL", iflab.thenlabel);
    }
    lower_push(iflab.elselabel);
    lower_push(iflab.endlabel);
    lower_push(STKIF);
    lower_end_stmt(std);
    break;

  case A_ENDIF:
    lower_start_stmt(lineno, label, TRUE, std);
    lower_check_stack(STKIF);
    iflab.endlabel = lower_pop();
    iflab.elselabel = lower_pop();
    plower("oL", "LABEL", iflab.endlabel);
    if (iflab.elselabel != iflab.endlabel) {
      plower("oL", "LABEL", iflab.elselabel);
    }
    lower_end_stmt(std);
    break;

  case A_GOTO:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("oS", "BR", A_SPTRG(A_L1G(ast)));
    lower_end_stmt(std);
    break;

  case A_MP_PDO:
/* cancel/cancellation */
    if (A_TASKLOOPG(ast)) {
      lower_do_stmt(std, ast, lineno, label); /* treat as normal do */
      break;
    }
    dotop = A_ENDLABG(ast);
    if (dotop) {
      dotop = A_SPTRG(dotop);
      lower_push(dotop);
    } else {
      lower_push(0);
    }
    lower_push(STKCANCEL);
    FLANG_FALLTHROUGH;
  case A_DO:
    lower_do_stmt(std, ast, lineno, label);
    break;

  case A_DOWHILE:
    lower_start_stmt(lineno, label, TRUE, std);
    dotop = lower_lab();
    dobottom = lower_lab();
    ++lowersym.docount;
    lower_push(0); /* no 'dotrip' variable */
    lower_push(dotop);
    lower_push(dobottom);
    lower_push(0); /*  used by openmp llvm */
    lower_push(0); /* no dovariable */
    lower_push(0); /* no 'doinc' variable */
    lower_push(0); /* schedtype */
    lower_push(STKDO);
    plower("oL", "LABEL", dotop);
    iflab.thenlabel = 0;
    iflab.elselabel = dobottom;
    iflab.endlabel = iflab.elselabel;
    lower_logical(A_IFEXPRG(ast), &iflab);
    if (iflab.thenlabel) {
      plower("oL", "LABEL", iflab.thenlabel);
    }
    lower_end_stmt(std);
    break;

  case A_ENDDO:
    lower_enddo_stmt(lineno, label, std, 0);
    break;
  case A_MP_ENDPDO:
    if (A_TASKLOOPG(ast))
      lower_enddo_stmt(lineno, label, std, 0);
    else
      lower_enddo_stmt(lineno, label, std, 1);
    break;

  case A_END:
    lower_start_stmt(lineno, label, FALSE, std);
    lower_finish_subprogram(gbl.rutype);
    if (gbl.rutype == RU_FUNC) {
      int fval = FVALG(gbl.currsub);
      ilm = plower("oS", "BASE", fval);
      switch (DTYG(DTYPEG(fval))) {
      case TY_BINT:
      case TY_SINT:
      case TY_INT:
      case TY_INT8:
      case TY_REAL:
      case TY_DBLE:
      case TY_QUAD:
      case TY_CMPLX:
      case TY_DCMPLX:
#ifdef TARGET_SUPPORTS_QUADFP
      case TY_QCMPLX:
#endif
      case TY_BLOG:
      case TY_SLOG:
      case TY_LOG:
      case TY_LOG8:
      case TY_WORD:
      case TY_DWORD:
        ilm = lower_typeload(DTYPEG(gbl.currsub), ilm);
        break;
      default:
        break;
      }
      plower("oi", "ENDF", ilm);
    } else {
      plower("o", "END");
    }
    lower_end_stmt(std);
    break;

  case A_PAUSE:
    /* won't see a 'pause' */
    lerror("unconverted PAUSE");
    break;

  case A_REDIM:
    src = A_SRCG(ast);
    if (A_TYPEG(src) == A_SUBSCR) {
      lop = A_LOPG(src);
      sptr = sym_of_ast(lop);
      lower_visit_symbol(sptr);
      if (A_TYPEG(lop) == A_ID) {
        sptr = A_SPTRG(lop);
      } else if (A_TYPEG(lop) == A_MEM) {
        sptr = A_SPTRG(A_MEMG(lop));
      } else {
        lerror("unsupported redim target");
        return;
      }
      /* generate the assignments, if any are needed */
      /* if the array is linearized, then the assignments are enough */
      if (LNRZDG(sptr))
        break;
      lower_start_stmt(lineno, label, TRUE, std);
      put_adjarr_bounds(DTYPEG(sptr), A_ASDG(src), NULL);
      lower_end_stmt(std);
    } else {
      ast_error("unexpected redimension argument", ast);
    }
    break;

  case A_RETURN:
    lower_start_stmt(lineno, label, TRUE, std);
    if (gbl.arets && A_LOPG(ast)) {
      /* alternate return */
      lower_expression(A_LOPG(ast));
      ilm = lower_conv(A_LOPG(ast), DT_INT4);
      plower("oi", "ARET", ilm);
    } else {
      plower("o", "RET");
    }
    lower_end_stmt(std);
    break;

  case A_STOP:
    /* won't see a 'stop' */
    lerror("unconverted STOP");
    break;

  case A_COMMENT:
  case A_COMSTR:
    /* ignore commented stuff */
    break;

  case A_BARRIER:
    lower_start_stmt(lineno, label, TRUE, std);
    symfunc = lower_makefunc(mkRteRtnNm(RTE_barrier), DT_NONE, FALSE);
    ilm = plower("onS", "CALL", 0, symfunc);
    lower_end_stmt(std);
    break;

  case A_ATOMIC:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "BEGINATOMIC");
    lower_end_stmt(std);
    break;

  case A_ATOMICCAPTURE:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "BEGINATOMICCAPTURE");
    lower_end_stmt(std);
    break;

  case A_ATOMICREAD:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "BEGINATOMICREAD");
    lower_end_stmt(std);
    break;

  case A_ATOMICWRITE:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "BEGINATOMICWRITE");
    lower_end_stmt(std);
    break;

  case A_ENDATOMIC:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "ENDATOMIC");
    lower_end_stmt(std);
    break;

  case A_MP_ATOMIC:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "MP_ATOMIC");
    lower_end_stmt(std);
    break;

  case A_MP_ENDATOMIC:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "MP_ENDATOMIC");
    lower_end_stmt(std);
    break;

  case A_MP_ATOMICWRITE:
    lower_start_stmt(lineno, label, TRUE, std);
    lower_omp_atomic_write(ast, lineno);
    lower_end_stmt(std);
    break;

  case A_MP_ATOMICUPDATE:
    lower_start_stmt(lineno, label, TRUE, std);
    lower_omp_atomic_update(ast, lineno);
    lower_end_stmt(std);
    break;

  case A_MP_ATOMICCAPTURE:
    lower_start_stmt(lineno, label, TRUE, std);
    lower_omp_atomic_capture(ast, lineno);
    lower_end_stmt(std);
    break;

  case A_MP_ATOMICREAD:
  case A_MP_CRITICAL:
    break;

  case A_MP_ENDCRITICAL:
    /*
     * Check for empty critical sections - without this check,
     * lower_end_stmt() is not called for
     *    critical
     *    endcritical
     *    endparallel
     * which of course is catastrophic since a mp_pexit call
     * is not generated.
     */
    for (prev = STD_PREV(std); prev; prev = STD_PREV(prev)) {
      int opc = A_TYPEG(STD_AST(prev));
      if (opc != A_COMMENT) {
        if (opc == A_MP_CRITICAL) {
          lower_start_stmt(lineno, label, TRUE, std);
          lower_end_stmt(prev);
        }
        break;
      }
    }
    break;

  case A_MP_PARALLEL:
    lowersym.parallel_depth++;
    lowersym.sc = SC_PRIVATE;
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = 0;
    flag = 0;
    if (A_IFPARG(ast) == 0) {
      ilm = plower("oS", "ICON", lowersym.intzero);
    } else {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
      ilm = plower("oi", "LNOT", ilm);
      flag = flag | 0x04;
    }

    if (A_NPARG(ast) == 0) {
      lilm = plower("oS", "ICON", lowersym.intzero);
    } else {
      lower_expression(A_NPARG(ast));
      lilm = lower_conv(A_NPARG(ast), DT_LOG4);
      flag = flag | 0x02;
    }
    proc_bind = 0;
    if (A_PROCBINDG(ast)) {
      proc_bind = get_int_cval(A_SPTRG((A_PROCBINDG(ast))));
      flag = flag | 0x01;
    }
    /* <logical if> <num_threads> <flag value> <proc_bind value> */
    ilm = plower("oiinn", "BPARA", ilm, lilm, flag, proc_bind);

    /* cancel/cancellation */
    dotop = A_ENDLABG(ast);
    if (dotop) {
      dotop = A_SPTRG(dotop);
      lower_push(dotop);
    } else {
      lower_push(0);
    }
    lower_push(STKCANCEL);

    lower_end_stmt(std);
    break;

  case A_MP_TEAMS:
    lowersym.parallel_depth++;
    lowersym.sc = SC_PRIVATE;
    lower_start_stmt(lineno, label, TRUE, std);
    if (A_THRLIMITG(ast) || A_NTEAMSG(ast)) {
      if (A_NTEAMSG(ast) == 0) {
        ilm = plower("oS", "ICON", lowersym.intone);
      } else {
        lower_expression(A_NTEAMSG(ast));
        ilm = lower_conv(A_NTEAMSG(ast), DT_INT);
      }
      if (A_THRLIMITG(ast) == 0) {
        lilm = plower("oS", "ICON", lowersym.intzero);
      } else {
        lower_expression(A_THRLIMITG(ast));
        lilm = lower_conv(A_THRLIMITG(ast), DT_INT);
      }
      ilm = plower("oii", "BTEAMSN", ilm, lilm);
    } else {
      ilm = plower("o", "BTEAMS");
    }
    lower_end_stmt(std);
    break;

  case A_MP_BMPSCOPE:
    stblk = A_STBLKG(ast);
    stblk = A_SPTRG(stblk);
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("oS", "BMPSCOPE", stblk);
    lower_end_stmt(std);
    break;
  case A_MP_EMPSCOPE:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "EMPSCOPE");
    lower_end_stmt(std);
    break;

  case A_MP_BARRIER:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "BARRIER");
    lower_end_stmt(std);
    break;

  case A_MP_TASKGROUP:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "TASKGROUP");
    lower_end_stmt(std);
    break;

  case A_MP_ETASKGROUP:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "ETASKGROUP");
    lower_end_stmt(std);
    break;

  case A_MP_TASKWAIT:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "TASKWAIT");
    lower_end_stmt(std);
    break;

  case A_MP_TASKYIELD:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "TASKYIELD");
    lower_end_stmt(std);
    break;

  case A_MP_TASKFIRSTPRIV:
    lop = A_SPTRG(A_LOPG(ast));
    rop = A_SPTRG(A_ROPG(ast));
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("oSS", "TASKFIRSTPRIV", lop, rop);
    lower_end_stmt(std);
    break;

  case A_MP_ENDPARALLEL:
    lower_start_stmt(lineno, label, TRUE, std);

    /* cancel/cancellation */
    lower_check_stack(STKCANCEL);
    dotop = lower_pop();
    if (dotop) {
      plower("oL", "LABEL", dotop);
    }

    plower("o", "EPAR");
    lower_end_stmt(std);
    --lowersym.parallel_depth;
    if (lowersym.parallel_depth == 0 && lowersym.task_depth == 0)
      lowersym.sc = SC_LOCAL;
    break;

  case A_MP_SINGLE:
    lower_start_stmt(lineno, label, TRUE, std);
    lab = lower_lab();
    ilm = plower("oS", "ICON", lowersym.intone);
    ilm = plower("oiS", "SINGLE", ilm, lab);
    lower_end_stmt(std);
    lower_push(lab);
    lower_push(STKSINGLE);
    break;

  case A_MP_ENDSINGLE:
    lower_check_stack(STKSINGLE);
    lab = lower_pop();
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("oL", "ESINGLE", lab);
    lower_end_stmt(std);
    break;

  case A_MP_MASTER:
    lower_start_stmt(lineno, label, TRUE, std);
    lab = lower_lab();
    ilm = plower("oS", "MASTER", lab);
    lower_end_stmt(std);
    lower_push(lab);
    lower_push(STKMASTER);
    break;

  case A_MP_ENDMASTER:
    lower_check_stack(STKMASTER);
    lab = lower_pop();
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("oL", "EMASTER", lab);
    lower_end_stmt(std);
    break;

  case A_MP_SECTIONS:
    lab = lower_lab();
    lower_start_stmt(lineno, label, TRUE, std);

    /* cancel/cancellation */
    dotop = A_ENDLABG(ast);
    if (dotop) {
      dotop = A_SPTRG(dotop);
      lower_push(dotop);
    } else {
      lower_push(0);
    }
    lower_push(STKCANCEL);

    ilm = plower("oS", "BSECTIONS", lab);
    lower_end_stmt(std);
    secnum = 0;
    lower_push(lab);    /* label */
    lower_push(secnum); /* section number */
    lower_push(STKSECTION);
    break;

  case A_MP_LSECTION:
    {
      lower_check_stack(STKSECTION);
      secnum = lower_pop();
      lab = lower_pop();
      nlab = lower_lab();
      lower_start_stmt(lineno, label, TRUE, std);
      ilm = plower("oS", "ICON", lower_getintcon(secnum));
      ilm = plower("oisL", "LSECTION", ilm, nlab, lab);
      lower_end_stmt(std);
      lower_push(nlab);
      lower_push(secnum + 1);
      lower_push(STKSECTION);
    }
    break;
  case A_MP_SECTION:
    {
      lower_check_stack(STKSECTION);
      secnum = lower_pop();
      lab = lower_pop();
      nlab = lower_lab();
      lower_start_stmt(lineno, label, TRUE, std);
      ilm = plower("oS", "ICON", lower_getintcon(secnum));
      ilm = plower("oisL", "SECTION", ilm, nlab, lab);
      lower_end_stmt(std);
      lower_push(nlab);
      lower_push(secnum + 1);
      lower_push(STKSECTION);
    }
    break;

  case A_MP_ENDSECTIONS:
    lower_check_stack(STKSECTION);
    secnum = lower_pop();
    lab = lower_pop();

    /* cancel/cancellation */
    lower_check_stack(STKCANCEL);
    dotop = lower_pop();

    lower_start_stmt(lineno, label, TRUE, std);
    if (dotop) {
      plower("oL", "LABEL", dotop);
    }
    ilm = plower("oL", "ESECTIONS", lab);
    lower_end_stmt(std);
    break;

  case A_MP_WORKSHARE:
    /* comment only, ignore */
    break;
  case A_MP_ENDWORKSHARE:
    lower_start_stmt(lineno, label, TRUE, std);
    lower_end_stmt(std);
    break;

  case A_MP_TASK:
  case A_MP_TASKLOOP:
    lowersym.task_depth++;
    lowersym.sc = SC_PRIVATE;
    lower_start_stmt(lineno, label, TRUE, std);
    /*
     *  num (bitvector):
     *   0x01 -- untied
     *   0x02 -- if clause present
     *   0x04 -- orphaned (dynamic, not lexically, parallel)
     *   0x08 -- nested task
     *   0x10 -- forced defer (CUDA)
     *   0x20 -- final task
     *   0x40 -- execute immediately
     *   0x80 -- mergeable
     */
    num = 0;
    if (A_UNTIEDG(ast))
      num |= 1; /* untied was specified */
    if (A_MERGEABLEG(ast))
      num |= MP_TASK_MERGEABLE;
    if (A_IFPARG(ast) == 0) {
      ilm = plower("oS", "ICON", lowersym.intone);
    } else {
      num |= 2; /* if clause is present */
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
    }
    ilm2 = plower("oS", "ICON", lowersym.intzero);
    if (A_FINALPARG(ast)) {
      num |= MP_TASK_FINAL;
      lower_expression(A_FINALPARG(ast));
      ilm2 = lower_conv(A_FINALPARG(ast), DT_LOG4);
    }
    if (A_PRIORITYG(ast)) {
      lower_expression(A_PRIORITYG(ast));
      ilm3 = lower_conv(A_PRIORITYG(ast), DT_INT4);
    } else {
      ilm3 = plower("oS", "ICON", lowersym.intzero);
    }

    if (A_TYPEG(ast) == A_MP_TASK) {
      if (A_EXEIMMG(ast))
        num |= 0x40;
      lab = lower_lab();
      ilm = plower("oSnii", "BTASK", lab, num, ilm, ilm2);
    } else {
      if (A_EXEIMMG(ast))
        num |= MP_TASK_IMMEDIATE;
      if (A_NOGROUPG(ast))
        num |= MP_TASK_NOGROUP;
      if (A_GRAINSIZEG(ast)) {
        num |= MP_TASK_GRAINSIZE;
        lower_expression(A_GRAINSIZEG(ast));
        ilm4 = lower_conv(A_GRAINSIZEG(ast), DT_INT4);
      } else if (A_NUM_TASKSG(ast)) {
        num |= MP_TASK_NUM_TASKS;
        lower_expression(A_NUM_TASKSG(ast));
        ilm4 = lower_conv(A_NUM_TASKSG(ast), DT_INT4);
      } else {
        ilm4 = plower("oS", "ICON", lowersym.intzero);
      }
      lab = lower_lab();
      ilm = plower("oSniiii", "BTASKLOOP", lab, num, ilm, ilm2, ilm3, ilm4);
    }
    lower_end_stmt(std);
    lower_push(lab); /* label */
    lower_push(STKTASK);
    lowersym.sc = SC_PRIVATE;
    if (A_TYPEG(ast) == A_MP_TASKLOOP) {
      break;
    }

    /* cancel/cancellation */
    dotop = A_ENDLABG(ast);
    if (dotop) {
      dotop = A_SPTRG(dotop);
      lower_push(dotop);
    } else {
      lower_push(0);
    }
    lower_push(STKCANCEL);

    break;

  case A_MP_TASKDUP:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "BTASKDUP");
    lower_end_stmt(std);
    break;
  case A_MP_ETASKDUP:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "ETASKDUP");
    lower_end_stmt(std);
    break;

  case A_MP_TASKREG:
    lowersym.sc = SC_PRIVATE;

    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "TASKREG");
    lower_end_stmt(std);
    break;

  case A_MP_TASKLOOPREG: {
    int lb, lbast, ub, ubast, st, stast;
    lower_start_stmt(lineno, label, TRUE, std);

    lbast = A_M1G(ast);
    ubast = A_M2G(ast);
    stast = A_M3G(ast);

    ilm = plower("o", "TASKLOOPVARS");

    lower_expression(lbast);
    lb = lower_ilm(lbast);
    lb = lower_conv_ilm(lbast, lb, A_NDTYPEG(lbast), DT_INT8);
    lower_reinit();

    lower_expression(ubast);
    ub = lower_ilm(ubast);
    ub = lower_conv_ilm(ubast, ub, A_NDTYPEG(ubast), DT_INT8);
    lower_reinit();

    if (stast == 0) {
      stast = astb.k1;
    }
    if (A_ALIASG(stast))
      stast = A_ALIASG(stast);
    lower_expression(stast);
    st = lower_ilm(stast);
    st = lower_conv_ilm(stast, st, A_NDTYPEG(stast), DT_INT8);
    lower_reinit();

    ilm = plower("oiii", "TASKLOOPREG", lb, ub, st);
    lower_end_stmt(std);
    lowersym.sc = SC_PRIVATE;
  } break;

  case A_MP_ETASKLOOPREG:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "ETASKLOOPREG");
    lower_end_stmt(std);
    break;

  case A_MP_TARGETDATA:
    lower_start_stmt(lineno, label, TRUE, std);
    if (A_IFPARG(ast)) {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
    } else {
      ilm = plower("oS", "ICON", lowersym.intone);
    }
    ilm = plower("oi", "BTARGETDATA", ilm);
    lower_end_stmt(std);
    break;

  case A_MP_ENDTARGETDATA:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "ETARGETDATA");
    lower_end_stmt(std);
    break;

  case A_MP_TARGETENTERDATA:
    lower_start_stmt(lineno, label, TRUE, std);
    flag = 0;
    if (A_IFPARG(ast)) {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
      flag |= MP_TGT_IFTARGET;
    } else {
      ilm = plower("oS", "ICON", lowersym.intone);
    }
    ilm = plower("oin", "TARGETENTERDATA", ilm, flag);
    lower_end_stmt(std);
    break;

  case A_MP_TARGETEXITDATA:
    lower_start_stmt(lineno, label, TRUE, std);
    flag = 0;
    if (A_IFPARG(ast)) {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
      flag |= MP_TGT_IFTARGET;
    } else {
      ilm = plower("oS", "ICON", lowersym.intone);
    }
    ilm = plower("oin", "TARGETEXITDATA", ilm, flag);
    lower_end_stmt(std);
    break;

  case A_MP_TARGETUPDATE:
    lower_start_stmt(lineno, label, TRUE, std);
    flag = 0;
    if (A_IFPARG(ast)) {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
      flag |= MP_TGT_IFTARGET;
    } else {
      ilm = plower("oS", "ICON", lowersym.intone);
    }
    ilm = plower("oin", "BTARGETUPDATE", ilm, flag);
    lower_end_stmt(std);
    break;

  case A_MP_TARGET:
    lower_start_stmt(lineno, label, TRUE, std);
    flag = 0;
    if (A_IFPARG(ast)) {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
      flag |= MP_TGT_IFTARGET;
    } else {
      ilm = plower("oS", "ICON", lowersym.intone);
    }

    if(flg.omptarget) {
      int ilm_numteams=0, ilm_numthreads=0, ilm_threadlimit=0;
      if(A_NTEAMSG(ast)) {
        lower_expression(A_NTEAMSG(ast));
        ilm_numteams = lower_conv(A_NTEAMSG(ast), DT_INT);
      } else {
        ilm_numteams = plower("oS", "ICON", lowersym.intzero);
      }
      if(A_THRLIMITG(ast)) {
        lower_expression(A_THRLIMITG(ast));
        ilm_threadlimit = lower_conv(A_THRLIMITG(ast), DT_INT);
      } else {
        ilm_threadlimit = plower("oS", "ICON", lowersym.intzero);
      }
      if(A_NPARG(ast)) {
        lower_expression(A_NPARG(ast));
        ilm_numthreads = lower_conv(A_NPARG(ast), DT_INT);
      } else {
        ilm_numthreads = plower("oS", "ICON", lowersym.intzero);
      }
      if(A_LOOPTRIPCOUNTG(ast) != 0) {
        lower_omp_target_tripcount(A_LOOPTRIPCOUNTG(ast), std);
      }
      plower("oniii", "MP_TARGETMODE", A_COMBINEDTYPEG(ast), ilm_numteams,
          ilm_threadlimit, ilm_numthreads);
    }

    //pragmatype specifies combined type of target.
    ilm = plower("oin", "BTARGET", ilm, flag);
    lower_end_stmt(std);
    break;
  case A_MP_MAP:
      lower_start_stmt(lineno, label, TRUE, std);
      lop = A_LOPG(ast);
      lower_expression(lop);
      //todo ompaccel need to pass size and base
      flag = A_PRAGMATYPEG(STD_AST(std));
      plower("oin", "MP_MAP", lower_base(lop), flag);
      lower_end_stmt(std);
    break;
  case A_MP_BREDUCTION:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "MP_BREDUCTION");
    lower_end_stmt(std);
    break;
  case A_MP_EREDUCTION:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "MP_EREDUCTION");
    lower_end_stmt(std);
    break;
  case A_MP_REDUCTIONITEM:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("ossn", "MP_REDUCTIONITEM", A_SHSYMG(ast), A_PRVSYMG(ast), A_REDOPRG(ast));
    lower_end_stmt(std);
    break;
  case A_MP_EMAP:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "MP_EMAP");
    lower_end_stmt(std);
    break;
  case A_MP_ENDTARGET:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "ETARGET");
    lower_end_stmt(std);
    break;

  case A_MP_ENDTEAMS:
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("o", "ETEAMS");
    lower_end_stmt(std);
    break;

  case A_MP_DISTRIBUTE:
    break;

  case A_MP_ENDDISTRIBUTE:
    break;

  case A_MP_ETASKLOOP:
    --lowersym.task_depth;
    if (lowersym.parallel_depth == 0 && lowersym.task_depth == 0)
      lowersym.sc = SC_LOCAL;
    lower_check_stack(STKTASK);
    lab = lower_pop();
    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("oL", "ETASKLOOP", lab);
    lower_end_stmt(std);
    break;

  case A_MP_ENDTASK:
    /* cancel/cancellation */
    lower_check_stack(STKCANCEL);
    dotop = lower_pop();

    lower_start_stmt(lineno, label, TRUE, std);
    if (dotop) {
      plower("oL", "LABEL", dotop);
    }
    lower_end_stmt(std);

    --lowersym.task_depth;
    if (lowersym.parallel_depth == 0 && lowersym.task_depth == 0)
      lowersym.sc = SC_LOCAL;
    lower_check_stack(STKTASK);
    lab = lower_pop();

    lower_start_stmt(lineno, label, TRUE, std);
    ilm = plower("oL", "ETASK", lab);
    lower_end_stmt(std);
    break;

  case A_MP_BCOPYIN:
    /* look for A_MP_COPYIN statements following this one */
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "BCOPYIN");
    for (nextstd = STD_NEXT(std); nextstd; nextstd = STD_NEXT(nextstd)) {
      int nextast;
      nextast = STD_AST(nextstd);
      if (nextast == 0 || A_TYPEG(nextast) != A_MP_COPYIN)
        break;
      sptr = A_SPTRG(nextast);
      if (!ALLOCATTRG(sptr)) {
        plower("oS", "COPYIN", sptr);
      } else {
        lower_expression(A_ROPG(nextast));
        rilm = lower_ilm(A_ROPG(nextast));
        plower("oSi", "COPYIN_A", sptr, rilm);
      }
    }
    plower("o", "ECOPYIN");
    lower_end_stmt(std);
    break;

  case A_MP_BCOPYPRIVATE:
    /* look for A_MP_COPYPRIVATE statements following this one */
    lower_start_stmt(lineno, label, TRUE, std);
    lilm = plower("oS", "ICON", lowersym.intone);
    ilm = plower("oi", "BCOPYPRIVATE", lilm);
    for (nextstd = STD_NEXT(std); nextstd; nextstd = STD_NEXT(nextstd)) {
      int nextast;
      nextast = STD_AST(nextstd);
      if (nextast == 0 || A_TYPEG(nextast) != A_MP_COPYPRIVATE)
        break;
      sptr = A_SPTRG(nextast);
      if (SCG(sptr) == SC_PRIVATE || SCG(sptr) == SC_DUMMY ||
          SCG(sptr) == SC_LOCAL) {
        ilm = plower("oS", "BASE", sptr);
        plower("oii", "COPYPRIVATE_P", lilm, ilm);
      } else if (SCG(sptr) == SC_BASED) {
        ilm = plower("oS", "BASE", MIDNUMG(sptr));
        if (!ALLOCATTRG(sptr)) {
          plower("oii", "COPYPRIVATE_P", lilm, ilm);
        } else {
          lower_expression(A_ROPG(nextast));
          rilm = lower_ilm(A_ROPG(nextast));
          plower("oiii", "COPYPRIVATE_PA", lilm, ilm, rilm);
        }
      } else {
        plower("oiS", "COPYPRIVATE", lilm, sptr);
      }
    }
    plower("oi", "ECOPYPRIVATE", lilm);
    lower_end_stmt(std);
    break;

  case A_MP_COPYIN:
  case A_MP_ECOPYIN:
  case A_MP_COPYPRIVATE:
  case A_MP_ECOPYPRIVATE:
    /* all work done at the A_MP_BCOPYIN/A_MP_BCOPYPRIVATE */
    break;

  case A_PREFETCH:
    lower_start_stmt(lineno, label, TRUE, std);
    lower_expression(A_LOPG(ast));
    ilm = lower_base(A_LOPG(ast));
    plower("oin", "PREFETCH", ilm, A_OPTYPEG(ast));
    lower_end_stmt(std);
    break;
  case A_PRAGMA:
    break;

  case A_MP_BPDO:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "BPDO");
    lower_end_stmt(std);
    break;

  case A_MP_EPDO:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "EPDO");
    lower_end_stmt(std);
    break;

  case A_MP_CANCEL:
    lower_start_stmt(lineno, label, TRUE, std);
    if (A_IFPARG(ast) == 0) {
      ilm = plower("oS", "ICON", lowersym.intone);
    } else {
      lower_expression(A_IFPARG(ast));
      ilm = lower_conv(A_IFPARG(ast), DT_LOG4);
    }
    sptr = A_SPTRG(A_ENDLABG(ast));
    num = A_CANCELKINDG(ast);
    plower("oSni", "CANCEL", sptr, num, ilm);
    lower_end_stmt(std);
    break;

  case A_MP_CANCELLATIONPOINT:
    lower_start_stmt(lineno, label, TRUE, std);
    sptr = A_SPTRG(A_ENDLABG(ast));
    num = A_CANCELKINDG(ast);
    plower("oSn", "CANCELPOINT", sptr, num);
    lower_end_stmt(std);
    break;

  case A_MP_BORDERED:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "MPBORDERED");
    lower_end_stmt(std);
    break;

  case A_MP_EORDERED:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "MPEORDERED");
    lower_end_stmt(std);
    break;

  case A_MP_FLUSH:
    lower_start_stmt(lineno, label, TRUE, std);
    plower("o", "FLUSH");
    lower_end_stmt(std);
    break;

    /* ------------- unsupported AST types ------------- */

  case A_CRITICAL:
  case A_ELSEFORALL:
  case A_ELSEWHERE:
  case A_ENDCRITICAL:
  case A_ENDFORALL:
  case A_ENDMASTER:
  case A_ENDWHERE:
  case A_FORALL:
  case A_HALB:
  case A_HAUB:
  case A_HGLB:
  case A_HGUB:
  case A_HEXTENT:
  case A_HALLOBNDS:
  case A_HSECT:
  case A_HARRAY:
  case A_HDISTR:
  case A_HNEWPROC:
  case A_HLOCALIZEBNDS:
  case A_HBLOCKLOOP:
  case A_HCYCLICLP:
  case A_HOFFSET:
  case A_HCOPYIN:
  case A_HCOPYOUT:
  case A_HCOPYSCLR:
  case A_HGETSCLR:
  case A_HNPROCS:
  case A_HFREE:
  case A_HFREEN:
  case A_HISLOCAL:
  case A_HLOCALOFFSET:
  case A_HCOPYSECT:
  case A_HPERMUTESECT:
  case A_HOVLPSHIFT:
  case A_HGATHER:
  case A_HSCATTER:
  case A_HCSTART:
  case A_HCFINISH:
  case A_HCFREE:
  case A_HOWNERPROC:
  case A_MASTER:
  case A_NOBARRIER:
  case A_REALIGN:
  case A_REDISTRIBUTE:
  case A_TRIPLE:
  case A_WHERE:
    ast_error("unsupported ast optype in statement", ast);
    break;
  default:
    ast_error("unknown ast optype in statement", ast);
    break;
  }
  lower_reset_temps();
} /* lower_stmt */

/* Return the ilm representing the number of elements of an array; only
 * called when the value can be extracted from the NUMELM field of the
 * array's dtype record.
 * A return value of 0 indicates an error occurred.
 */
static int
lower_numelm(int dtype, int ast, int src, int loadit)
{
  int numilm, numelm;
  int nelm;

  numelm = ADD_NUMELM(dtype);
  nelm = numelm;
  if (numelm == 0) {
    ast_error("allocate with no array size", ast);
    return 0;
  }
  if (size_of(A_DTYPEG(numelm)) < 8) {
    numelm = mk_convert(numelm, DT_INT8);
  }
  numelm_constant = 0;
  if (A_TYPEG(numelm) == A_CNST) {
    numilm = plower("oS", lowersym.bnd.con, A_SPTRG(numelm));
    numelm_constant = numelm;
  } else if (A_TYPEG(numelm) == A_ID) {
    if (STYPEG(A_SPTRG(numelm)) == ST_MEMBER) {
      lerror("number-of-elements symbol %s is a member reference",
             SYMNAME(A_SPTRG(numelm)));
    }
    numilm = plower("oS", "BASE", A_SPTRG(numelm));
    if (loadit)
      numilm = lower_typeload(DTYPEG(A_SPTRG(numelm)), numilm);
  } else {
    if (nelm == numelm) {
      numelm = check_member(src, numelm);
    } else {
      numelm = check_member(src, nelm);
      numelm = mk_convert(numelm, DT_INT8);
    }
    lower_expression(numelm);
    numilm = lower_ilm(numelm);
  }

  return numilm;
}

/* generate code to refer to a symbol;
 * set 'pointerval' to:
 *  VarBase to get a base address suitable for dereferencing;
 *  SourceBase to get a 'source' address for pointer assignment;
 *  TargetBase to get a 'target' address for pointer assignment;
 *  ArgumentBase to get an 'address' for argument passing. */
static int
lower_sptr(int sptr, int pointerval)
{
  int base;
  assert(sptr > NOSYM, "lower_sptr: bad sptr", sptr, ERR_Severe);
  if (LOWER_SYMBOL_REPLACE(sptr)) {
    sptr = LOWER_SYMBOL_REPLACE(sptr);
  }
  lower_visit_symbol(sptr);
  if (SCG(sptr) == SC_BASED) {
    int midnum;
    if (ALLOCG(sptr) && MIDNUMG(sptr) == 0) {
      fill_midnum(sptr);
    }
    midnum = MIDNUMG(sptr);
    while (SCG(midnum) == SC_BASED) {
      /* multiple derefs are treated as a single deref */
      midnum = MIDNUMG(midnum);
    }

    base = plower("oS", "BASE", midnum);
    switch (pointerval) {
    case VarBase:
    case ArgumentBase:
      if (XBIT(70, 4) && !lower_disable_ptr_chk) {
        lower_check_pointer(mk_id(sptr), base);
      }
      base = plower("oiS", "PLD", base, sptr);
      ADDRTKNP(sptr, 1);
      break;
    case SourceBase:
      base = lower_typeload(DT_ADDR, base);
      break;
    case TargetBase:
      break;
    }
  } else if (ALLOCG(sptr) || POINTERG(sptr)) {
    int dtype = DTYPEG(sptr);
    if (DTY(dtype) != TY_PTR) {
      if (MIDNUMG(sptr) == 0) {
        if (STYPEG(sptr) == ST_MEMBER) {

          lerror("allocatable pointer symbol %s is a member reference",
                 SYMNAME(sptr));
        }
        base = plower("oS", "BASE", sptr);
      } else {
        if (STYPEG(MIDNUMG(sptr)) == ST_MEMBER) {
          lerror("pointer symbol %s is a member reference",
                 SYMNAME(MIDNUMG(sptr)));
        }
        base = plower("oS", "BASE", MIDNUMG(sptr));
        switch (pointerval) {
        case SourceBase:
          base = lower_typeload(DT_ADDR, base);
          break;
        case VarBase:
        case ArgumentBase:
          if (XBIT(70, 4) && !lower_disable_ptr_chk) {
            lower_check_pointer(mk_id(sptr), base);
          }
          base = plower("oiS", "PLD", base, sptr);
          break;
        case TargetBase:
          break;
        }
      }
    } else {
      /* scalar pointer */
      if (STYPEG(sptr) == ST_MEMBER) {
        lerror("scalar pointer symbol %s is a member reference", SYMNAME(sptr));
      }
      base = plower("oS", "BASE", sptr);
      switch (pointerval) {
      case SourceBase:
        base = lower_typeload(DT_ADDR, base);
        break;
      case VarBase:
      case ArgumentBase:
        if (XBIT(70, 4) && !lower_disable_ptr_chk) {
          lower_check_pointer(mk_id(sptr), base);
        }
        base = plower("oirC", "PLD", base, 0, sptr);
        break;
      case TargetBase:
        break;
      }
    }
  } else {
    if (STYPEG(sptr) == ST_MEMBER) {
      /* special case this error: user has referenced an array
         with the wrong number of array indices */
      if (SYMNAME(sptr) && strstr(SYMNAME(sptr), "$sd") != 0)
        lerror("symbol %s is an inconsistent array descriptor", SYMNAME(sptr));
      else
        lerror("symbol %s is a member reference", SYMNAME(sptr));
    }
    base = plower("oS", "BASE", sptr);
    switch (pointerval) {
    case SourceBase:
      base = plower("oi", "LOC", base);
      break;
    case VarBase:
    case TargetBase:
    case ArgumentBase:
      break;
    }
  }
  return base;
} /* lower_sptr */

/* get base address for LHS */
static int
lower_base_address(int ast, int pointerval)
{
  int base = 0, lbase, sptr, ilm, lilm, rilm, lop, save_disable_ptr_chk;
  switch (A_TYPEG(ast)) {
  case A_ID:
    base = lower_sptr(A_SPTRG(ast), pointerval);
    break;
  case A_MEM:
    sptr = A_SPTRG(A_MEMG(ast));
    lower_visit_symbol(sptr);
    save_disable_ptr_chk = lower_disable_ptr_chk;
    lower_disable_ptr_chk = 0;
    lbase = lower_ilm(A_PARENTG(ast));
    lbase = intermediate_members(lbase, A_PARENTG(ast), sptr);
    lower_disable_ptr_chk = save_disable_ptr_chk;
    switch (pointerval) {
    case VarBase:
    case ArgumentBase:
      if (POINTERG(sptr) || (ALLOCG(sptr) && MIDNUMG(sptr))) {
        base = plower("oiS", "MEMBER", lbase, MIDNUMG(sptr));
        if (XBIT(70, 4) && !lower_disable_ptr_chk) {
          lower_check_pointer(ast, base);
        }
        base = plower("oirC", "PLD", base, sptr, sptr);
      } else
        base = plower("oiS", "MEMBER", lbase, sptr);
      break;
    case SourceBase:
      if (POINTERG(sptr) || (ALLOCG(sptr) && MIDNUMG(sptr))) {
        base = plower("oiS", "MEMBER", lbase, MIDNUMG(sptr));
        base = lower_typeload(DT_ADDR, base);
      } else {
        base = plower("oiS", "MEMBER", lbase, sptr);
        base = plower("oi", "LOC", base);
      }
      break;
    case TargetBase:
      if (POINTERG(sptr) || (ALLOCG(sptr) && MIDNUMG(sptr))) {
        base = plower("oiS", "MEMBER", lbase, MIDNUMG(sptr));
      } else {
        base = plower("oiS", "MEMBER", lbase, sptr);
      }
      break;
    }
    break;
  case A_SUBSTR:
    save_disable_ptr_chk = lower_disable_ptr_chk;
    lower_disable_ptr_chk = 0;
    ilm = lower_base(A_LOPG(ast));
    if (A_LEFTG(ast))
      lilm = lower_ilm(A_LEFTG(ast));
    else
      lilm = plower("oS", "ICON", lowersym.intone);
    if (A_RIGHTG(ast))
      rilm = lower_ilm(A_RIGHTG(ast));
    else {
      int len;
      int lop = A_LOPG(ast);
      len = DTY(A_NDTYPEG(lop) + 1); /* char string length */
      if (len && A_ALIASG(len)) {
        len = A_ALIASG(len);
        len = A_SPTRG(len);
        rilm = plower("oS", "ICON", len); /* ilm */
      } else {
        /* assumed length string, use 'len' function */
        rilm = plower("oi", "LEN", ilm);
      }
    }
    if (DTY(A_NDTYPEG(ast)) == TY_NCHAR) {
      base = plower("oiii", "NSUBS", ilm, lilm, rilm);
    } else {
      base = plower("oiii", "SUBS", ilm, lilm, rilm);
    }
    switch (pointerval) {
    case SourceBase:
      base = plower("oi", "LOC", base);
      FLANG_FALLTHROUGH;
    case VarBase:
    case TargetBase:
    case ArgumentBase:
      break;
    }
    lower_disable_ptr_chk = save_disable_ptr_chk;
    break;
  case A_SUBSCR:
    save_disable_ptr_chk = lower_disable_ptr_chk;
    lower_disable_ptr_chk = 0;
    base = lower_base(ast);
    switch (pointerval) {
    case SourceBase:
      base = plower("oi", "LOC", base);
      break;
    case VarBase:
    case TargetBase:
    case ArgumentBase:
      break;
    }
    lower_disable_ptr_chk = save_disable_ptr_chk;
    break;
  case A_CNST:
    switch (DTYG(A_NDTYPEG(ast))) {
    case TY_HOLL:
      ilm = A_ILMG(ast);
      base = plower("oi", "DPREF", ilm);
      A_BASEP(ast, base);
      break;
    default:
      ast_error("unknown constant type for LHS or argument", ast);
      break;
    }
    break;
  case A_CONV:
    save_disable_ptr_chk = lower_disable_ptr_chk;
    lower_disable_ptr_chk = 0;
    lop = A_LOPG(ast);
    base = lower_base(lop);
    lower_disable_ptr_chk = save_disable_ptr_chk;
    break;
  default:
    ast_error("unknown operator for LHS or argument", ast);
    break;
  }
  return base;
} /* lower_base_address */

/** \brief Get base address for LHS. */
int
lower_base(int ast)
{
  int base;
  base = A_BASEG(ast);
  if (base == 0 && A_TYPEG(ast) == A_SUBSCR) {
    /* to prevent infinite recursion, call lower_base_address()
     * on base address of the subscript.
     */
    base = lower_base_address(A_LOPG(ast), VarBase);
    A_BASEP(ast, base);
  } else if (base == 0) {
    base = lower_base_address(ast, VarBase);
    A_BASEP(ast, base);
  }
  return base;
} /* lower_base */

/** \brief Get base address for LHS for a symbol. */
int
lower_base_sptr(int sptr)
{
  int base;
  base = lower_sptr(sptr, VarBase);
  return base;
} /* lower_base_sptr */

/** \brief Get source address for pointer assignment. */
int
lower_address(int ast)
{
  return lower_base_address(ast, SourceBase);
} /* lower_address */

/** \brief Get target address for pointer assignment. */
int
lower_target(int ast)
{
  return lower_base_address(ast, TargetBase);
} /* lower_target */

int
lower_typeload(int dtype, int base)
{
  int ilm;
  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_BLOG:
    ilm = plower("oi", "CHLD", base);
    break;
  case TY_SINT:
    ilm = plower("oi", "SILD", base);
    break;
  case TY_INT:
    ilm = plower("oi", "ILD", base);
    break;
  case TY_INT8:
    ilm = plower("oi", "KLD", base);
    break;
  case TY_SLOG:
    ilm = plower("oi", "SLLD", base);
    break;
  case TY_LOG:
    ilm = plower("oi", "LLD", base);
    break;
  case TY_LOG8:
    ilm = plower("oi", "KLLD", base);
    break;
  case TY_REAL:
    ilm = plower("oi", "RLD", base);
    break;
  case TY_DBLE:
    ilm = plower("oi", "DLD", base);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  /* output the quad precision load in ilm */
  case TY_QUAD:
    ilm = plower("oi", "QFLD", base);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oi", "CLD", base);
    break;
  case TY_DCMPLX:
    ilm = plower("oi", "CDLD", base);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oi", "CQLD", base);
    break;
#endif
  case TY_PTR:
    if (!XBIT(49, 0x20000000)) {
      ilm = plower("oir", "PLD", base, 0);
    } else if (XBIT(49, 0x100)) { /* 64-bit pointers */
      ilm = plower("oi", "KLD", base);
    } else {
      ilm = plower("oi", "ILD", base);
    }
    break;
  case TY_CHAR:
  case TY_NCHAR:
  case TY_DERIVED:
  case TY_STRUCT:
  case TY_UNION:
  case TY_ARRAY:
    ilm = base;
    break;
  default:
    ast_error("unexpected data type at load", dtype);
    ilm = base;
    break;
  }
  return ilm;
} /* lower_typeload */

int
lower_typestore(int dtype, int base, int rhs)
{
  int ilm;
  switch (DTY(dtype)) {
  case TY_BINT:
  case TY_BLOG:
    ilm = plower("oii", "CHST", base, rhs);
    break;
  case TY_SINT:
    ilm = plower("oii", "SIST", base, rhs);
    break;
  case TY_INT:
    ilm = plower("oii", "IST", base, rhs);
    break;
  case TY_INT8:
    ilm = plower("oii", "KST", base, rhs);
    break;
  case TY_SLOG:
    ilm = plower("oii", "SLST", base, rhs);
    break;
  case TY_LOG:
    ilm = plower("oii", "LST", base, rhs);
    break;
  case TY_LOG8:
    ilm = plower("oii", "KLST", base, rhs);
    break;
  case TY_REAL:
    ilm = plower("oii", "RST", base, rhs);
    break;
  case TY_DBLE:
    ilm = plower("oii", "DST", base, rhs);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QUAD:
    ilm = plower("oii", "QFST", base, rhs);
    break;
#endif
  case TY_CMPLX:
    ilm = plower("oii", "CST", base, rhs);
    break;
  case TY_DCMPLX:
    ilm = plower("oii", "CDST", base, rhs);
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case TY_QCMPLX:
    ilm = plower("oii", "CQST", base, rhs);
    break;
#endif
  case TY_PTR:
    if (!XBIT(49, 0x20000000)) {
      ilm = plower("oii", "PST", base, rhs);
    } else if (XBIT(49, 0x100)) { /* 64-bit pointers */
      ilm = plower("oii", "KST", base, rhs);
    } else {
      ilm = plower("oii", "IST", base, rhs);
    }
    break;
  case TY_CHAR:
  case TY_NCHAR:
    ilm = plower("oii", "CHST", base, rhs);
    break;
  case TY_DERIVED:
  case TY_STRUCT:
  case TY_UNION:
  case TY_ARRAY:
    ilm = base;
    break;
  default:
    ast_error("unexpected data type at store", dtype);
    ilm = base;
    break;
  }
  return ilm;
} /* lower_typestore */

/** \brief Get the ilm for this expression ast. */
int
lower_ilm(int ast)
{
  int ilm, base, lop, sptr, dtype;
  ilm = A_ILMG(ast);
  if (ilm == 0) {
    switch (A_TYPEG(ast)) {
    case A_ID:
      base = lower_base(ast);
      sptr = A_SPTRG(ast);
      if (PTRVG(sptr) && !XBIT(49, 0x20000000) && HCCSYMG(sptr)) {
        ilm = lower_typeload(DT_ADDR, base);
      } else {
        if (!NDTYPE_IS_SET(ast)) {
          A_NDTYPEP(ast, A_DTYPEG(ast));
        }
        dtype = A_NDTYPEG(ast);
        ilm = lower_typeload(dtype, base);
      }
      break;
    case A_MEM:
      base = lower_base(ast);
      if (PTRVG(A_SPTRG(A_MEMG(ast))) && !XBIT(49, 0x20000000)) {
        ilm = lower_typeload(DT_ADDR, base);
      } else {
        if (!NDTYPE_IS_SET(ast)) {
          A_NDTYPEP(ast, A_DTYPEG(ast));
        }
        ilm = lower_typeload(A_NDTYPEG(ast), base);
      }
      break;
    case A_SUBSCR:
      base = lower_base(ast);
      if (!NDTYPE_IS_SET(ast)) {
        A_NDTYPEP(ast, A_DTYPEG(ast));
      }
      ilm = lower_typeload(A_NDTYPEG(ast), base);
      break;
    case A_CONV:
      /* should only reach here if no conversion was needed anyway */
      lop = A_LOPG(ast);
      ilm = lower_ilm(lop);
      FLANG_FALLTHROUGH;
    default:
      break;
    }
    A_ILMP(ast, ilm);
  }
  return ilm;
} /* lower_ilm */

static int symone;

static VAR *
export_data_varlist_asts(VAR *ivl)
{
  do {
    if (ivl->u.varref.subt) {
      export_data_varlist_asts(ivl->u.varref.subt);
    } else {
      lower_expression(ivl->u.varref.ptr);
      lower_base(ivl->u.varref.ptr);
    }
    ivl = ivl->next;
  } while (ivl != NULL && ivl->id == Varref);
  return ivl;
} /* export_data_varlist_asts */

static void
export_data_const_asts(ACL *ict)
{
  int ast;
  for (; ict != NULL; ict = ict->next) {
    if (ict->id == AC_IEXPR) {
      export_data_const_asts(ict->u1.expr->lop);
      if (ict->u1.expr->rop) {
        export_data_const_asts(ict->u1.expr->rop);
      }
    } else if (!ict->subc) {
      switch (DTY(ict->dtype)) {
      case TY_REAL:
      case TY_INT:
        /* simply put out the value */
        break;
      default:
        ast = ict->u1.ast;
        if (ast && A_TYPEG(ast) != A_INIT && A_ALIASG(ast)) {
          ast = A_ALIASG(ast);
        } else {
          ast = astb.i1;
        }
        lower_visit_symbol(A_SPTRG(ast));
        break;
      }
    } else {
      export_data_const_asts(ict->subc);
    }
  }
} /* export_data_const_asts */

static void
export_data_asts(VAR *ivl, ACL *ict)
{
  if (ivl == NULL && ict->subc != NULL)
    return; /* structures? */
  if (!ivl) {
    lower_visit_symbol(ict->sptr);
  } else {
    VAR *next;
    for (; ivl != NULL; ivl = next) {
      next = ivl->next;
      switch (ivl->id) {
      case Dostart:
        lower_expression(ivl->u.dostart.indvar);
        lower_ilm(ivl->u.dostart.indvar);
        lower_expression(ivl->u.dostart.lowbd);
        lower_expression(ivl->u.dostart.upbd);
        if (ivl->u.dostart.step) {
          lower_expression(ivl->u.dostart.step);
        } else {
          if (symone == 0) {
            symone = plower("oS", "ICON", lowersym.intone);
          }
        }
        break;
      case Doend:
        break;
      case Varref:
        next = export_data_varlist_asts(ivl);
        break;
      default:
        break;
      }
    }
  }
  export_data_const_asts(ict);
} /* export_data_asts */

static VAR *
export_data_varlist(VAR *ivl)
{
  int ast;
  do {
    if (ivl->u.varref.subt) {
      export_data_varlist(ivl->u.varref.subt);
    } else {
      ast = ivl->u.varref.ptr;
      if (A_TYPEG(ast) == A_ID) {
        fprintf(lower_ilm_file, "Variable i%d t%d\n", A_BASEG(ast),
                A_NDTYPEG(ast));
      } else {
        fprintf(lower_ilm_file, "Reference i%d t%d\n", A_BASEG(ast),
                A_NDTYPEG(ast));
      }
      lower_use_datatype(A_NDTYPEG(ast), 1);
    }
    ivl = ivl->next;
  } while (ivl != NULL && ivl->id == Varref);
  return ivl;
} /* export_data_varlist */

static void
export_data_consts(ACL *ict, int spoof)
{
  int repeatc, ast, sptr, dtype;
  for (; ict != NULL; ict = ict->next) {
    if (ict->repeatc) {
      ast = ict->repeatc;
      if (A_ALIASG(ast)) {
        ast = A_ALIASG(ast);
      } else {
        lerror("DATA repeat count ast not constant: %d", ast);
        ast = astb.i1;
      }
      sptr = A_SPTRG(ast);
      repeatc = CONVAL2G(sptr);
    } else {
      repeatc = 1;
    }
    if (ict->id == AC_IEXPR) {
      /* operator, repeat, datatype, sptr */
      lower_use_datatype(ict->dtype, 1);
      fprintf(lower_ilm_file, "Constant EXPR n%d n%d s%d t%d\n", repeatc,
              ict->u1.expr->op, ict->sptr, ict->dtype);
      fprintf(lower_ilm_file, "Constant OPERAND\n");
      export_data_consts(ict->u1.expr->lop, 0);
      fprintf(lower_ilm_file, "Constant operandend\n");
      if (ict->u1.expr->rop) {
        fprintf(lower_ilm_file, "Constant OPERAND\n");
        export_data_consts(ict->u1.expr->rop, 0);
        fprintf(lower_ilm_file, "Constant operandend\n");
      }
    } else if (ict->id == AC_IDO) {
      /* idx_var_sptr */
      ACL *wict;
      fprintf(lower_ilm_file, "Constant DO s%d\n", ict->u1.doinfo->index_var);
      lower_visit_symbol(ict->u1.doinfo->index_var);

      fprintf(lower_ilm_file, "Constant BOUNDS\n");
      wict =
          construct_acl_from_ast(ict->u1.doinfo->init_expr, astb.bnd.dtype, 0);
      export_data_consts(wict, 0);
      fprintf(lower_ilm_file, "Constant boundsend\n");
      fprintf(lower_ilm_file, "Constant BOUNDS\n");
      wict =
          construct_acl_from_ast(ict->u1.doinfo->limit_expr, astb.bnd.dtype, 0);
      export_data_consts(wict, 0);
      fprintf(lower_ilm_file, "Constant boundsend\n");
      fprintf(lower_ilm_file, "Constant BOUNDS\n");
      wict =
          construct_acl_from_ast(ict->u1.doinfo->step_expr, astb.bnd.dtype, 0);
      export_data_consts(wict, 0);
      fprintf(lower_ilm_file, "Constant boundsend\n");

      DOVARP(ict->u1.doinfo->index_var, 1);
      export_data_consts(ict->subc, 0);
      DOVARP(ict->u1.doinfo->index_var, 0);
      fprintf(lower_ilm_file, "Constant doend\n");
    } else if (ict->id == AC_ACONST) {
      /* datatype */
      lower_use_datatype(ict->dtype, 1);
      fprintf(lower_ilm_file, "Constant ARRAY s%d t%d\n", ict->sptr,
              ict->dtype);
      export_data_consts(ict->subc, 0);
      fprintf(lower_ilm_file, "Constant arrayend\n");
    } else if (ict->id == AC_SCONST) {
      /* repeat count, datatype */
      lower_use_datatype(ict->dtype, 1);
      fprintf(lower_ilm_file, "structure n%d t%d s%d n%d\n", repeatc,
              ict->dtype, ict->sptr, ict->no_dinitp);
      export_data_consts(ict->subc, 0);
      fprintf(lower_ilm_file, "tructurend\n");
    } else if (ict->id == AC_ICONST) {
      fprintf(lower_ilm_file, "Constant LITRLINT n%d\n", ict->u1.i);
    } else {
      if (ict->id == AC_AST || !ict->subc) {
        if (ict->sptr > 0)
          lower_visit_symbol(ict->sptr);
        if (ict->sptr > 0 && spoof) {
          int ilm;
          ilm = plower("oS", "BASE", ict->sptr);
          fprintf(lower_ilm_file, "Variable i%d t%d\n", ilm, ict->dtype);
          ict->sptr = 0;
        }
        ast = ict->u1.ast;
        if (A_TYPEG(ast) == A_CNST) {
          /* repeat count, datatype */
          dtype = ict->ptrdtype ? ict->ptrdtype : ict->dtype;
          fprintf(lower_ilm_file, "Constant CONSTANT n%d t%d s%d", repeatc,
                  dtype, ict->sptr);
          lower_use_datatype(dtype, 1);
          if (A_ALIASG(ast)) {
            ast = A_ALIASG(ast);
          } else if (!PARAMG(A_SPTRG(ast)) && !DOVARG(A_SPTRG(ast))) {
            lerror("INIT ast not constant: %d", ast);
            ast = astb.i1;
          }
          sptr = A_SPTRG(ast);
          switch (DTY(ict->dtype)) {
          case TY_REAL:
          case TY_BINT:
          case TY_SINT:
          case TY_INT:
          case TY_BLOG:
          case TY_SLOG:
          case TY_LOG:
          case TY_WORD:
            fprintf(lower_ilm_file, " n%d", CONVAL2G(sptr));
            break;
          case TY_HOLL:
            /* sptr = CONVAL1G( sptr ); holl fix */
            fprintf(lower_ilm_file, " s%d", sptr);
            lower_visit_symbol(sptr);
            break;
          case TY_NCHAR:
            sptr = CONVAL1G(sptr);
            fprintf(lower_ilm_file, " s%d", sptr);
            lower_visit_symbol(sptr);
            break;
          default:
            fprintf(lower_ilm_file, " s%d", sptr);
            lower_visit_symbol(sptr);
            break;
          }
          fprintf(lower_ilm_file, "\n");
        } else if (A_TYPEG(ast) == A_CONV) {
          ACL *wict;
          wict = construct_acl_from_ast(ast, A_DTYPEG(ast), 0);
          export_data_consts(wict, 0);
        } else if (A_TYPEG(ast) == A_SUBSCR) {
          ACL *wict;
          wict = construct_acl_from_ast(ast, A_DTYPEG(ast), 0);
          export_data_consts(wict, 0);
        } else if (A_TYPEG(ast) == A_BINOP) {
          ACL *wict;
          wict = construct_acl_from_ast(ast, A_DTYPEG(ast), 0);
          export_data_consts(wict, 0);
        } else {
          /* repeat count, datatype sptr */
          fprintf(lower_ilm_file, "Constant ID n%d t%d s%d s%d\n", repeatc,
                  ict->dtype, A_SPTRG(ast), ict->sptr);
          lower_use_datatype(ict->dtype, 1);
          if (A_ALIASG(ast)) {
            ast = A_ALIASG(ast);
          } else {
            ast = ict->u1.ast;
          }
          lower_visit_symbol(A_SPTRG(ast));
        }
      } else {
        /* repeat count, datatype */
        lower_use_datatype(ict->dtype, 1);
        fprintf(lower_ilm_file, "structure n%d t%d s%d n%d\n", repeatc,
                ict->dtype, ict->sptr, ict->no_dinitp);
        export_data_consts(ict->subc, 0);
        if (ict->sptr) {
          lower_visit_symbol(ict->sptr);
        }
        fprintf(lower_ilm_file, "tructurend\n");
      }
    }
  }
} /* export_data_consts */

static void
export_data_stmt(VAR *ivl, ACL *ict)
{
  if (!ivl) {
    export_data_consts(ict, 1);
  } else {
    VAR *next;
    for (; ivl != NULL; ivl = next) {
      next = ivl->next;
      switch (ivl->id) {
      case Dostart:
        fprintf(lower_ilm_file, "Do i%d i%d i%d", A_ILMG(ivl->u.dostart.indvar),
                A_ILMG(ivl->u.dostart.lowbd), A_ILMG(ivl->u.dostart.upbd));
        if (ivl->u.dostart.step) {
          fprintf(lower_ilm_file, " i%d", A_ILMG(ivl->u.dostart.step));
        } else {
          fprintf(lower_ilm_file, " i%d", symone);
        }
        fprintf(lower_ilm_file, "\n");
        break;
      case Doend:
        fprintf(lower_ilm_file, "Enddo\n");
        break;
      case Varref:
        next = export_data_varlist(ivl);
        break;
      default:
        break;
      }
    }
    export_data_consts(ict, 0);
  }
} /* export_data_stmt */

#define DTIO_RSPTR 1
#define DTIO_WSPTR 2
#define DTIO_DTVSPTR 3
#define DTIO_DTV_SDSC 4
#define DTIO_VLIST_SPTR 5
#define DTIO_VLIST_SDSC 6
#define DTIO_END 7

void
lower_data_stmts(void)
{
  int nw, lineno, fileno, cnt, astplist, in_defined_io, tsptr, fsptr;
  VAR *ivl; /* variable list */
  ACL *ict; /* constant tree */
  DREC *drec;

  int flg_x70; /* save flg.x[70], which holds -Mchkptr and -Mbounds */

  flg_x70 = flg.x[70];
  flg.x[70] &= ~0x7; /* reset XBIT(70,[124]) */

  if (astb.df != NULL && (flg.ipa & 0x20)) {
    nw = fseek(astb.df, 0L, 0);
    if (nw == -1)
      perror("lower_data_stmts - fseek on astb.df");
    while (1) {
      nw = fread(&lineno, sizeof(lineno), 1, astb.df);
      if (nw == 0)
        break;
      if (nw != 1) {
        error(0, 4, 0, "catastrophic error lowering data statements",
              "reading lineno");
      }
      nw = fread(&fileno, sizeof(fileno), 1, astb.df);
      if (nw == 0)
        break;
      if (nw != 1) {
        error(0, 4, 0, "catastrophic error lowering data statements",
              "reading fileno");
      }
      nw = fread(&ivl, sizeof(ivl), 1, astb.df);
      if (nw == 0)
        break;
      if (nw != 1) {
        error(0, 4, 0, "catastrophic error lowering data statements",
              "reading ivl");
      }
      nw = fread(&ict, sizeof(ict), 1, astb.df);
      if (nw != 1) {
        error(0, 4, 0, "catastrophic error lowering data statements",
              "reading ict");
      }

      /* If the ivl is a pointer whose storage class in not static, we
       *  skip it (it will be initialized by lower_pointer_init),
       */
      if (ivl && ivl->u.varref.id == S_IDENT) {
        int sptr = A_SPTRG(ivl->u.varref.ptr);
        if (!XBIT(7, 0x100000) && !DINITG(sptr))
          continue;

        /*
         * Emit dinit info for procedure pointers.
         *
         * DO NOT emit dinit info for regular POINTERs -- delete as part of
         * the fix to place init'd pointers in the STATIC# container
         * (data section)
         */
        if (!is_procedure_ptr(sptr) && (SCG(sptr) == SC_BASED ||
            (SCG(sptr) == SC_CMBLK && POINTERG(sptr)))) {
          continue;
        }
      }

      fprintf(lower_ilm_file, "Begindata\n");
      /* dump out the ASTs used for this initializer */
      lower_start_stmt(lineno, 0, FALSE, 0);
      /* dump ASTs in loops, etc. */
      symone = 0;
      export_data_asts(ivl, ict);
      export_data_stmt(ivl, ict);
      ast_unvisit_norepl();
      fprintf(lower_ilm_file, "Writedata\n");
    }
  }

  /* now lower already processed data initialization, such as formats...*/
  dinit_fseek(0);
  in_defined_io = 0;
  while ((drec = dinit_read())) {
    int charlen, okaddr;
    int dinittype = drec->dtype;
    int dinitval = drec->conval;

    switch (dinittype) {
    case DINIT_END:
      /* end of a dinit record */
      fprintf(lower_ilm_file, "Init end");
      fprintf(lower_ilm_file, "\n");
      break;
    case DINIT_LABEL:
      ++cnt;
      /* this is a namelist label */
      if (in_defined_io) {
        /* the order is from semfin.c -- nml_emit_desc*/
        switch (in_defined_io) {
        case DTIO_RSPTR:
          fsptr = dinitval; /* read function */
        read_write_func:
          if (dinitval == 0) {
            if (DT_PTR == DT_INT8) {
              int v[4], sptr;
              v[0] = v[1] = v[2] = v[3] = 0;
              sptr = getcon(v, DT_PTR);
              VISITP(sptr, 1);
              fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", sptr,
                      DT_PTR);
            } else if (size_of(DT_PTR) == size_of(DT_INT8)) {
              int v[4], sptr;
              v[0] = v[1] = v[2] = v[3] = 0;
              sptr = getcon(v, DT_PTR);
              VISITP(sptr, 1);
              fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", sptr,
                      DT_PTR);
            } else {
              fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", 0, DT_PTR);
            }
            ++in_defined_io;
            goto done_dinit_label;
          }
          break;
        case DTIO_WSPTR:
          if (fsptr == 0)
            fsptr = dinitval; /* write function */
          if (dinitval == 0) {
            goto read_write_func;
          }
          break;
        case DTIO_DTVSPTR:
          tsptr = dinitval; /* keep this for to get descriptor */
          if (MIDNUMG(dinitval))
            dinitval = MIDNUMG(dinitval);
          break;
        case DTIO_DTV_SDSC:
          dinitval = SDSCG(tsptr); /* dtv descriptor */
          if (dinitval == 0) {
            dinitval = get_type_descr_arg(fsptr, tsptr);
            assert(dinitval > 0, "dtv descriptor not found:", tsptr, 3);
          }
          break;
        case DTIO_VLIST_SPTR:
          break;
        case DTIO_VLIST_SDSC:
          break;
        default:
          tsptr = dinitval; /* keep this for next dinit_label */
          break;
        }
        ++in_defined_io;
        if (in_defined_io == 7)
          in_defined_io = 0;        /* done defined io */
      } else if (dinitval == -98) { /* start defined io */
        int sptr = 0;
        if (DT_PTR == DT_INT8) {
          int v[4], sptr;
          v[2] = v[3] = 0;
          v[0] = -1;
          v[1] = -98;
          sptr = getcon(v, DT_PTR);
          VISITP(sptr, 1);
          fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", sptr, DT_PTR);
        } else if (size_of(DT_PTR) == size_of(DT_INT8)) {
          int v[4], sptr;
          v[2] = v[3] = 0;
          v[0] = -1;
          v[1] = -98;
          sptr = getcon(v, DT_PTR);
          VISITP(sptr, 1);
          fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", sptr, DT_PTR);
        } else {
          fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", -98, DT_PTR);
        }
        dinitval = sptr;

        in_defined_io = 1;
        break;
      }
      lower_visit_symbol(dinitval);
      if (STYPEG(dinitval) == ST_CONST) {
        okaddr = 1;
      } else {
        okaddr = 0;
        {
          if (SCG(dinitval) == SC_BASED && ADJARRG(dinitval)) {
            /* automatic array; for now, semant will generate an assigment
             * of the array's pointer to the namelsit descriptor
             */
            fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", -99,
                    DT_INT4);
            if (size_of(DT_PTR) == size_of(DT_INT8))
              fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", -99,
                      DT_INT4);
            goto done_dinit_label;
          }
          if (SCG(dinitval) != SC_DUMMY) {
            if (SCG(dinitval) != SC_LOCAL)
              okaddr = 1;
            else if (SAVEG(dinitval) || DINITG(dinitval))
              okaddr = 1;
          }
        }
      }
      if (okaddr) {
        fprintf(lower_ilm_file, "Init Label:%d\n", dinitval);
      } else {
        int dest, src, subs[7], ent, ast;
        if (size_of(DT_PTR) == size_of(DT_INT8)) {
          int v[4], sptr;
          v[2] = v[3] = 0;
          v[0] = -1;
          v[1] = -99;
          sptr = getcon(v, DT_PTR);
          VISITP(sptr, 1);
          fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", sptr, DT_PTR);
        } else {
          fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", -99, DT_PTR);
        }
        lower_use_datatype(DT_INT, 1);
        lower_use_datatype(DT_PTR, 1);

        subs[0] = mk_cval(cnt, DT_INT);
        src = mk_id(dinitval);
        dest = mk_subscr(astplist, subs, 1, DT_PTR);
        src = mk_unop(OP_LOC, src, DT_PTR);
        ast = mk_assn_stmt(dest, src, DT_PTR);
        for (ent = gbl.currsub; ent != NOSYM; ent = SYMLKG(ent)) {
          int std;
          if (SCG(dinitval) == SC_DUMMY) {
            if (PTRVG(dinitval)) {
              /* get address of dummy descriptors */
              if (is_argp_in_entry(ent, dinitval)) {
                std = add_stmt_after(ast, ENTSTDG(ent));
                ENTSTDP(ent, std);
              }
            }
            if (is_arg_in_entry(ent, dinitval)) {
              std = add_stmt_after(ast, ENTSTDG(ent));
              ENTSTDP(ent, std);
            }
          } else { /* local variable */
            std = add_stmt_after(ast, ENTSTDG(ent));
            ENTSTDP(ent, std);
          }
        }
      }
    done_dinit_label:
      break;
    case DINIT_LOC:
      lower_visit_symbol(dinitval);
      fprintf(lower_ilm_file, "Init location:%d\n", dinitval);
      cnt = 0;
      break;
    case DINIT_FMT:
      lower_visit_symbol(dinitval);
      fprintf(lower_ilm_file, "Init format:%d\n", dinitval);
      astplist = mk_id(dinitval);
      cnt = 0;
      break;
    case DINIT_NML:
      lower_visit_symbol(dinitval);
      fprintf(lower_ilm_file, "Init namelist:%d\n", dinitval);
      astplist = mk_id(dinitval);
      cnt = 0;
      break;
    case DINIT_REPEAT:
      cnt += dinitval - 1;
      fprintf(lower_ilm_file, "Init repeat:%d\n", dinitval);
      break;
    default:
      ++cnt;
      switch (DTY(dinittype)) {
      case TY_BINT:
      case TY_SINT:
      case TY_INT:
      case TY_BLOG:
      case TY_SLOG:
      case TY_LOG:
      case TY_FLOAT:
        fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", dinitval,
                dinittype);
        lower_use_datatype(dinittype, 1);
        break;
      case TY_DBLE:
      case TY_QUAD:
      case TY_CMPLX:
      case TY_DCMPLX:
      case TY_QCMPLX:
      case TY_INT8:
      case TY_LOG8:
        fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", dinitval,
                dinittype);
        lower_visit_symbol(dinitval);
        lower_use_datatype(dinittype, 1);
        break;
      case TY_CHAR:
        dinittype = DTYPEG(dinitval);
        fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", dinitval,
                dinittype);
        lower_visit_symbol(dinitval);
        lower_use_datatype(dinittype, 1);
        /* if there are not an even number of words, pad out to
         * a multiple of four */
        charlen = string_length(DTYPEG(dinitval));
        charlen = charlen & 0x3;
        if (charlen) {
          charlen = 4 - charlen;
          while (charlen) {
            fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", 0, DT_BLOG);
            lower_use_datatype(DT_BLOG, 1);
            --charlen;
          }
        }
        break;
      case TY_NCHAR:
        dinittype = DTYPEG(dinitval);
        fprintf(lower_ilm_file, "Init symbol:%d datatype:%d\n", dinitval,
                dinittype);
        lower_use_datatype(dinittype, 1);
        lower_visit_symbol(dinitval);
        /* if there are not an even number of words, pad out to
         * a multiple of four */
        charlen = string_length(DTYPEG(dinitval));
        charlen = charlen & 0x1;
        if (charlen) {
          fprintf(lower_ilm_file, "Init value:%d datatype:%d\n", 0, DT_SLOG);
          lower_use_datatype(DT_SLOG, 1);
        }
        break;
      case TY_PTR:
        lerror("data initialization with pointer type");
        break;
      default:
        lerror("data initialization with unknown type");
        break;
      }
      break;

    case DINIT_STARTARY:
      fprintf(lower_ilm_file, "Init array start\n");
      break;
    case DINIT_ENDARY:
      fprintf(lower_ilm_file, "Init array end\n");
      break;
    case DINIT_TYPEDEF:
      fprintf(lower_ilm_file, "Init typedef start %d\n", dinitval);
      lower_visit_symbol(dinitval);
      break;
    case DINIT_ENDTYPE:
      fprintf(lower_ilm_file, "Init typedef end\n");
      break;
    case DINIT_STR:
      ++cnt;
      lower_visit_symbol(dinitval);
      fprintf(lower_ilm_file, "Init charstring:%d\n", dinitval);
      break;

    case 0:
    case DINIT_OFFSET:
    case DINIT_ZEROES:
      lerror("error in lowering data initialization");
      break;
    }
  }
  dinit_fseek_end();
  flg.x[70] = flg_x70; /* restore */
} /* lower_data_stmts */
