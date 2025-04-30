/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - inliner utils definitions
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "ast.h"
#include "gramtk.h"
#include "semant.h"
#include "optimize.h"
#include "interf.h"
#include "inliner.h"
#include "ccffinfo.h"
#include "fdirect.h"

/* Switches:
 *	-q 49 1:	Trace
 *	-q 49 4:	Dump flowgraph
 *	-q 49 64:	Dump STDs
 *	-x 49 0x200000: Inhibit inlining
 */

/* Macros: */
#if DEBUG
#define TRACE0(s)    \
  if (DBGBIT(49, 1)) \
  fprintf(gbl.dbgfil, s)
#define TRACE1(s, a1) \
  if (DBGBIT(49, 1))  \
  fprintf(gbl.dbgfil, s, a1)
#define TRACE2(s, a1, a2) \
  if (DBGBIT(49, 1))      \
  fprintf(gbl.dbgfil, s, a1, a2)
#define TRACE3(s, a1, a2, a3) \
  if (DBGBIT(49, 1))          \
  fprintf(gbl.dbgfil, s, a1, a2, a3)
#else
#define TRACE0(s)
#define TRACE1(s, a1)
#define TRACE2(s, a1, a2)
#define TRACE3(s, a1, a2, a3)
#endif

/* inliner version number must be updated whenever ILMs or symbol table
   format, or inliner interface change:  */
#define INLINER_VERSION 14

#define MAX_INLINE_NAME 42
#define MAX_FNAME_LEN 2048
#define TOC_HEADER_FMT "Inliner TOC V.%d"
#define TOC_ENTRY_FMT "%s %s %s %s"
#define MODNAME_FMT "i%d.e"

#define ERROR(s, v)  \
  {                  \
    interr(s, v, 1); \
    return;          \
  }
#define MAX_LINE_LEN 400
#define PERM_AREA 8
#define STASH(p, area) strcpy(getitem(area, strlen(p) + 1), p);

typedef struct fitem {   /* function item */
  char *sFunc;           /* name of function */
  struct fitem *pfiNext; /* pointer to next function item */
} FI;

typedef struct libentry {   /* info on one inliner library entry */
  char *sFunc;              /* user function name */
  char *sHost;              /* user host function name */
  char *sMod;               /* user containing module name */
  char *sModFile;           /* file containing encoded form */
  struct libentry *pleNext; /* pointer to next library entry */
} LE;

typedef struct argrep {/* argument replacement info */
  int sptrDummy;       /* dummy parameter */
  int astAct;          /* actual parameter */
} AR;

#define AR_SPTRDUMMY(i) parbase[i].sptrDummy
#define AR_ASTACT(i) parbase[i].astAct

/* Local functions: */
static void inline_stds(int stdStart, int stdLast, int iLevels, int level);
static int inline_ast(int std, int ast, int iLevels, int level);
static LOGICAL inline_func(int std, int ast, int iLevels, int level,
                           int *pEntry);
static void modify_inlined_stds(int sptrEntry, int ast, int stdstart,
                                int stdend);
static int find_entry(int sptrCall);
static int copy_inargs(int sptrEntry, int astCall, int stdStart, int stdEnd);
static void remove_inlined_stds();
static void remove_inlined_symbols(int);
static int promote_assumsz_arg(int sptrDummy, int astArg);
static void allocate_adjarrs(int stdstart, int stdend);
static void allocate_array(int sptr, int stdStart, int stdEnd);
static void remove_returns(int stdstart, int stdend);
static void move_labels(int stdstart, int stdlast);
static void rewrite_inlined_args(int astCall, int sptrEntry, int stdstart,
                                 int stdend);
static int replace_arg(int astDummy, int astAct);
static void assign_bounds(int sptrDummy, int astAct, int std);
static int get_subscr(int ast, unsigned int dim);
static void load_TOC(char *sDir);
static void store_funcname(char *sFunc);
static LE *find_libentry(const char *sMod, const char *sHost, char *sFunc);
static LOGICAL tkr_match_arg(int dtypDummy, int dtypAct);
static LOGICAL aliased_args(int sptrEntry, int astCall);
static LOGICAL make_arg_copy(int sptrEntry, int astCall, unsigned int arg);

/* Local data: */
static char *sExtDir = NULL; /* extract directory name */
static FI *pfiStart = NULL;  /* functions to be inlined */
static LE *pleStart = NULL;  /* table of contents */
static int sptrHigh = 0;     /* high-water mark of symbol table */
static int nLevels = 1;      /* # levels of calls to inline */
static AR *parbase;          /* parameter replacement table */
static int iarsize;          /* allocated size of parbase */
static int iaravail;         /* next available entry in parbase */
static int stdEnd;           /* STD # of END statement */

/*
 * Record the extract directory or a program unit name to extract.
 */
void
extractor_command_info(char *sDir, int nInsts, char *sFunc)
{
  if (sDir)
    load_TOC(sDir);
  if (sFunc)
    store_funcname(sFunc);
}

/*
 * Return TRUE if the current function can be extracted.
 */
LOGICAL
extractor_possible(void)
{
  char *sCurrFunc;
  FI *pfi;
  int sptr, std;
  int dtyp;

  /* Search for a named program unit. */
  if (pfiStart) {
    sCurrFunc = SYMNAME(gbl.currsub);
    for (pfi = pfiStart; pfi; pfi = pfi->pfiNext)
      if (!strcmp(pfi->sFunc, sCurrFunc))
        break;
    if (!pfi) {
      TRACE1("can't extract: program unit %s not named\n", sCurrFunc);
      return FALSE;
    }
  }
  if (gbl.rutype != RU_SUBR && gbl.rutype != RU_FUNC) {
    TRACE0("can't extract: program unit not subroutine or function\n");
    return FALSE;
  }
  if (SYMLKG(gbl.entries) != NOSYM) {
    TRACE0("can't extract: multiple entry points\n");
    ccff_info(MSGNEGINLINER, "INL024", gbl.findex, gbl.funcline,
              "%module%separator%function is not HL inlineable: multiple entry "
              "points",
              "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
              "separator=%s", gbl.currmod ? "::" : "", "function=%s",
              SYMNAME(gbl.currsub), NULL);
    return FALSE;
  }
  if (gbl.arets) {
    TRACE0("can't extract: multiple returns\n");
    ccff_info(
        MSGNEGINLINER, "INL025", gbl.findex, gbl.funcline,
        "%module%separator%function is not HL inlineable: alternate returns",
        "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "", "separator=%s",
        gbl.currmod ? "::" : "", "function=%s", SYMNAME(gbl.currsub), NULL);
    return FALSE;
  }
  for (sptr = stb.firstusym; sptr < stb.stg_avail; sptr++) {
    if (ST_ISVAR(STYPEG(sptr)) && SCOPEG(sptr) != stb.curr_scope)
      continue;
    if (STYPEG(sptr) == ST_NML) {
      TRACE0("can't extract: namelist present\n");
      ccff_info(MSGNEGINLINER, "INL026", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: namelist "
                "statements",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    }
    if (STYPEG(sptr) == ST_LABEL && FMTPTG(sptr)) {
      TRACE0("can't extract: format list present\n");
      ccff_info(
          MSGNEGINLINER, "INL027", gbl.findex, gbl.funcline,
          "%module%separator%function is not HL inlineable: format statements",
          "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "", "separator=%s",
          gbl.currmod ? "::" : "", "function=%s", SYMNAME(gbl.currsub), NULL);
      return FALSE;
    }
    if (ST_ISVAR(STYPEG(sptr)) && SCG(sptr) == SC_STATIC && SAVEG(sptr)) {
      TRACE0("can't extract: SAVEd local present\n");
      ccff_info(MSGNEGINLINER, "INL028", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: SAVEd local "
                "variables",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    }
    if (ST_ISVAR(STYPEG(sptr)) && DINITG(sptr)) {
      TRACE0("can't extract: data initialization present\n");
      ccff_info(MSGNEGINLINER, "INL029", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: DATA "
                "initialization",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    }
    if (ST_ISVAR(STYPEG(sptr)) && SCG(sptr) == SC_DUMMY && !IGNOREG(sptr)) {
      dtyp = DTYPEG(sptr);
      if (DTY(dtyp) == TY_CHAR || DTY(dtyp) == TY_NCHAR)
        if (dtyp == DT_ASSCHAR || dtyp == DT_ASSNCHAR || dtyp == DT_DEFERCHAR ||
            dtyp == DT_DEFERNCHAR) {
          TRACE0("can't extract: assumed/deferred length dummy present\n");
          ccff_info(MSGNEGINLINER, "INL030", gbl.findex, gbl.funcline,
                    "%module%separator%function is not HL inlineable: "
                    "assumed-length character dummy",
                    "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                    "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                    SYMNAME(gbl.currsub), NULL);
          return FALSE;
        }
    }
  }
  if (!XBIT(13, 0x200) && !INMODULEG(gbl.currsub)) {
    /* only extract module subprograms */
    return FALSE;
  }
  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    /* whether to allow loops or conditionals */
    switch (A_TYPEG(STD_AST(std))) {
    case A_DO:
      if (!XBIT(13, 0x100)) {
        ccff_info(MSGNEGINLINER, "INL031", gbl.findex, gbl.funcline,
                  "%module%separator%function is not HL inlineable: DO "
                  "statements disallowed",
                  "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                  "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                  SYMNAME(gbl.currsub), NULL);
        return FALSE;
      }
      break;
    case A_DOWHILE:
      ccff_info(MSGNEGINLINER, "INL032", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: DOWHILE "
                "statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_IF:
    case A_IFTHEN:
      if (!XBIT(13, 0x80)) {
        ccff_info(MSGNEGINLINER, "INL033", gbl.findex, gbl.funcline,
                  "%module%separator%function is not HL inlineable: IF "
                  "statements disallowed",
                  "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                  "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                  SYMNAME(gbl.currsub), NULL);
        return FALSE;
      }
      break;
    case A_AIF:
      ccff_info(MSGNEGINLINER, "INL034", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: arithmetic "
                "IF statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_GOTO:
      ccff_info(MSGNEGINLINER, "INL035", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: GOTO "
                "statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_CGOTO:
      ccff_info(MSGNEGINLINER, "INL036", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: computed "
                "GOTO statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_AGOTO:
      ccff_info(MSGNEGINLINER, "INL037", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: assigned "
                "GOTO statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_ASNGOTO:
      ccff_info(MSGNEGINLINER, "INL038", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: ASSIGN "
                "statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_STOP:
      ccff_info(MSGNEGINLINER, "INL039", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: STOP "
                "statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_PAUSE:
      ccff_info(MSGNEGINLINER, "INL040", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: PAUSE "
                "statements disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    case A_MP_PARALLEL:
      ccff_info(MSGNEGINLINER, "INL041", gbl.findex, gbl.funcline,
                "%module%separator%function is not HL inlineable: OpenMP "
                "parallel section disallowed",
                "module=%s", gbl.currmod ? SYMNAME(gbl.currmod) : "",
                "separator=%s", gbl.currmod ? "::" : "", "function=%s",
                SYMNAME(gbl.currsub), NULL);
      return FALSE;
    }
  }

  return TRUE;
} /* extractor_possible */

/*
 * Extract the current subroutine.
 */
void
extractor(void)
{
  FILE *fd;
  char *sCurrFunc;
  const char *sCurrHost, *sCurrMod;
  int iFile, nFile;
  LE *ple;
  int iStat;
  char sExtFile[MAX_FNAME_LEN];

  TRACE0("----------\n");
  TRACE0("Extractor:\n");
  if (!sExtDir) {
    errsev(270);
    return;
  }
  if (!extractor_possible())
    return;
  sCurrFunc = SYMNAME(gbl.currsub);
  if (gbl.internal <= 1 || gbl.outersub == 0) {
    sCurrHost = ".";
  } else {
    sCurrHost = SYMNAME(gbl.outersub);
  }
  if (gbl.currmod == 0) {
    sCurrMod = ".";
  } else {
    sCurrMod = SYMNAME(gbl.currmod);
  }

  /* Get the highest extract file #. */
  nFile = 0;
  for (ple = pleStart; ple; ple = ple->pleNext) {
    iStat = sscanf(ple->sModFile, MODNAME_FMT, &iFile);
    if (iStat != 1)
      return;
    if (iFile > nFile)
      nFile = iFile;
  }

  /* Search for an existing library entry. */
  ple = find_libentry(sCurrMod, sCurrHost, sCurrFunc);

  if (!ple) {
    /* Create a new library entry. */
    ple = (LE *)getitem(PERM_AREA, sizeof(LE));
    if (!ple)
      ERROR("extractor: can't allocate memory", 3);
    ple->sFunc = STASH(sCurrFunc, PERM_AREA);
    ple->sHost = STASH(sCurrHost, PERM_AREA);
    ple->sMod = STASH(sCurrMod, PERM_AREA);
    sprintf(sExtFile, MODNAME_FMT, nFile + 1);
    ple->sModFile = STASH(sExtFile, PERM_AREA);
    ple->pleNext = pleStart;
    pleStart = ple;
  }

  sprintf(sExtFile, "%s/%s", sExtDir, ple->sModFile);
  fd = fopen(sExtFile, "w");
  if (!fd) {
    error(275, 2, gbl.lineno, sExtFile, NULL);
    return;
  }
  export_inline(fd, sCurrFunc, sExtFile);
}

void
extractor_end(void)
{
  FILE *fd;
  char sTOCFile[MAX_FNAME_LEN];
  LE *ple;

  if (!sExtDir)
    return;

  /* Create the TOC file. */
  fndpath("TOC", sTOCFile, MAX_FNAME_LEN, sExtDir);

  fd = fopen(sTOCFile, "w");
  if (!fd) {
    error(274, 2, gbl.lineno, sExtDir, NULL);
    return;
  }

  /* Write the header. */
  fprintf(fd, TOC_HEADER_FMT, INLINER_VERSION);
  fprintf(fd, "\n");

  /* Write out the library entries. */
  for (ple = pleStart; ple; ple = ple->pleNext) {
    fprintf(fd, TOC_ENTRY_FMT, ple->sMod, ple->sHost, ple->sFunc,
            ple->sModFile);
    fprintf(fd, "\n");
  }
  fclose(fd);

  freearea(PERM_AREA);
}

void
inline_add_lib(char *sDir)
{
  load_TOC(sDir);
}

void
inline_add_func(char *sFunc, int n)
{
  if (sFunc)
    store_funcname(sFunc);
  else
    nLevels = n;
}

static int ninlined = 0;
static int nskip = 0;
/*
 * inline into this subprogram
 */
void
inliner(void)
{
  if (flg.x[115])
    nLevels = flg.x[115];

  if (!sExtDir)
    return;

  ninlined = 0;
  nskip = 0;

  if (XBIT(49, 0x200000))
    return;
  inline_stds(STD_NEXT(0), STD_PREV(0), nLevels, 1);

#if DEBUG
  if (DBGBIT(49, 64)) {
    fprintf(gbl.dbgfil, "----------- STDs after inlining\n");
    dump_std();
  }
#endif
}

/*
 * Perform inlining on all STDs between stdStart & stdLast repeating
 * recursively for iLevels.
 */
static void
inline_stds(int stdStart, int stdLast, int iLevels, int level)
{
  int std, stdNext;
  int ast, astNew;

  if (!iLevels)
    return;

  if (stdLast)
    stdLast = STD_NEXT(stdLast);
  for (std = stdStart; std != stdLast; std = stdNext) {
    stdNext = STD_NEXT(std);
    if (STD_LINENO(std))
      gbl.lineno = STD_LINENO(std);
    ast = STD_AST(std);
    astNew = inline_ast(std, ast, iLevels, level);
    if (astNew == ast)
      continue;
    if (!astNew) {
      delete_stmt(std);
      continue;
    }
    STD_AST(std) = astNew;
    A_STDP(astNew, std);
  }
}

/*
 * Inline any function calls within ast, and repeat on the inlined statements
 * iLevels times. Add inlined statements prior to
 * std. Replace any inlined call with a new temp. After replacement, return
 * the new AST. Return 0 if std will be deleted after inlining.
 */
static int
inline_ast(int std, int ast, int iLevels, int level)
{
  int astNewl, astNewr, astNew, vastSubs[7], astNewu, astNews, astNewc;
  int asd, sptrEntry;
  int nsubs, sub;
  int sptrRet;
  INT argt, argtNew;
  int arg, nargs;
  LOGICAL bChanged;

  if (ast == 0)
    return 0;
  /* Explicit recursion is used here instead of a call to ast_traverse()
   * because ast_traverse() is called by rewrite_inlined_args() to replace
   * dummy parameters with actual. */
  switch (A_TYPEG(ast)) {
  case A_NULL:
  case A_ID:
  case A_CNST:
  case A_LABEL:
  case A_CMPLXC:
  case A_MEM:
    return ast;
  case A_BINOP:
    astNewl = inline_ast(std, A_LOPG(ast), iLevels, level);
    astNewr = inline_ast(std, A_ROPG(ast), iLevels, level);
    if (astNewl == A_LOPG(ast) && astNewr == A_ROPG(ast))
      return ast;
    astNew = mk_binop(A_OPTYPEG(ast), astNewl, astNewr, A_DTYPEG(ast));
    return astNew;
  case A_UNOP:
    astNewl = inline_ast(std, A_LOPG(ast), iLevels, level);
    if (astNewl == A_LOPG(ast))
      return ast;
    astNew = mk_unop(A_OPTYPEG(ast), astNewl, A_DTYPEG(ast));
    return astNew;
  case A_CONV:
    astNewl = inline_ast(std, A_LOPG(ast), iLevels, level);
    if (astNewl == A_LOPG(ast))
      return ast;
    astNew = mk_convert(astNewl, A_DTYPEG(ast));
    return astNew;
  case A_PAREN:
    astNewl = inline_ast(std, A_LOPG(ast), iLevels, level);
    if (astNewl == A_LOPG(ast))
      return ast;
    astNew = mk_paren(astNewl, A_DTYPEG(ast));
    return astNew;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    nsubs = ASD_NDIM(asd);
    bChanged = FALSE;
    for (sub = 0; sub < nsubs; sub++) {
      vastSubs[sub] = inline_ast(std, ASD_SUBS(asd, sub), iLevels, level);
      bChanged |= (vastSubs[sub] != ASD_SUBS(asd, sub));
    }
    if (!bChanged)
      return ast;
    astNew = mk_subscr(A_LOPG(ast), vastSubs, nsubs, A_DTYPEG(ast));
    return astNew;
  case A_SUBSTR:
    astNewl = inline_ast(std, A_LEFTG(ast), iLevels, level);
    astNewr = inline_ast(std, A_RIGHTG(ast), iLevels, level);
    if (astNewl == A_LEFTG(ast) && astNewr == A_RIGHTG(ast))
      return ast;
    astNew = mk_substr(A_LOPG(ast), astNewl, astNewr, A_DTYPEG(ast));
    return astNew;
  case A_TRIPLE:
    astNewl = inline_ast(std, A_LBDG(ast), iLevels, level);
    astNewu = inline_ast(std, A_UPBDG(ast), iLevels, level);
    astNews = inline_ast(std, A_STRIDEG(ast), iLevels, level);
    if (astNewl == A_LBDG(ast) && astNewu == A_UPBDG(ast) &&
        astNews == A_STRIDEG(ast))
      return ast;
    astNew = mk_triple(astNewl, astNewu, astNews);
    return astNew;
  case A_FUNC:
  case A_INTR:
  case A_CALL:
  case A_ICALL:
    argt = A_ARGSG(ast);
    nargs = A_ARGCNTG(ast);
    argtNew = mk_argt(nargs);
    bChanged = FALSE;
    for (arg = 0; arg < nargs; arg++) {
      ARGT_ARG(argtNew, arg) =
          inline_ast(std, ARGT_ARG(argt, arg), iLevels, level);
      if (ARGT_ARG(argtNew, arg) != ARGT_ARG(argt, arg))
        bChanged = TRUE;
    }
    if (bChanged == FALSE) {
      unmk_argt(nargs);
      astNew = ast;
    } else {
      astNew = mk_func_node(A_TYPEG(ast), A_LOPG(ast), nargs, argtNew);
      A_SHAPEP(astNew, A_SHAPEG(ast));
      A_DTYPEP(astNew, A_DTYPEG(ast));
      if (A_TYPEG(ast) == A_INTR || A_TYPEG(ast) == A_ICALL) {
        A_OPTYPEP(astNew, A_OPTYPEG(ast));
      }
    }
    if (A_TYPEG(ast) == A_INTR || A_TYPEG(ast) == A_ICALL) {
      return astNew;
    }
#if DEBUG
    if (flg.x[17] != 0 && ninlined >= flg.x[17])
      return astNew;
#endif
    if (A_TYPEG(ast) == A_CALL) {
      bChanged = inline_func(std, astNew, iLevels, level, &sptrEntry);
      if (bChanged)
        return 0;
      else
        return astNew;
    }
    bChanged = inline_func(std, astNew, iLevels, level, &sptrEntry);
    if (!bChanged)
      return astNew;
    sptrRet = FVALG(sptrEntry);
    assert(sptrRet, "inline_ast: function value not found", std, 4);
    SCP(sptrRet, SC_LOCAL); /* ensures declaration's generation */
    astNew = mk_id(sptrRet);
    return astNew;
  case A_ASN:
    astNewr = inline_ast(std, A_SRCG(ast), iLevels, level);
    astNewl = inline_ast(std, A_DESTG(ast), iLevels, level);
    if (astNewr == A_SRCG(ast) && astNewl == A_DESTG(ast))
      return ast;
    astNew = mk_assn_stmt(astNewl, astNewr, A_DTYPEG(ast));
    return astNew;
  case A_IF:
  case A_IFTHEN:
    astNewl = inline_ast(std, A_IFEXPRG(ast), iLevels, level);
    if (astNewl == A_IFEXPRG(ast))
      return ast;
    astNew = mk_stmt(A_TYPEG(ast), 0);
    A_IFEXPRP(astNew, astNewl);
    A_IFSTMTP(astNew, A_IFSTMTG(ast));
    return astNew;
  case A_DO:
    astNewl = inline_ast(std, A_M1G(ast), iLevels, level);
    astNewr = inline_ast(std, A_M2G(ast), iLevels, level);
    astNews = inline_ast(std, A_M3G(ast), iLevels, level);
    astNewc = inline_ast(std, A_M4G(ast), iLevels, level);
    if (astNewl == A_M1G(ast) && astNewr == A_M2G(ast) &&
        astNews == A_M3G(ast) && astNewc == A_M4G(ast))
      return ast;
    astNew = mk_stmt(A_TYPEG(ast), 0);
    A_DOLABP(astNew, A_DOLABG(ast));
    A_DOVARP(astNew, A_DOVARG(ast));
    A_M1P(astNew, astNewl);
    A_M2P(astNew, astNewr);
    A_M3P(astNew, astNews);
    A_M4P(astNew, astNewc);
    return astNew;
  default:
    return ast;
  }
}

/*
 * if we inline foo into bar, and foo calls ugh, and bar has another
 * symbol for ugh, reuse bar's symbol for ugh instead
 */
static int use_old_subprogram_limits[2] = {0, 0};
static void
use_old_subprograms(int oldsymavl)
{
  int sptr, tptr;
  use_old_subprogram_limits[0] = oldsymavl;
  use_old_subprogram_limits[1] = stb.stg_avail - 1;
  for (sptr = oldsymavl; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_PROC:
      /* see if there is an identical function in the caller */
      for (tptr = HASHLKG(sptr); tptr > NOSYM; tptr = HASHLKG(tptr)) {
        if (NMPTRG(tptr) != NMPTRG(sptr))
          continue;
        if (STYPEG(tptr) != ST_PROC)
          continue;
        if (tptr >= sptrHigh)
          continue;
        if (INMODULEG(tptr) != INMODULEG(sptr))
          continue;
        if (INMODULEG(tptr)) {
          if (NMPTRG(ENCLFUNCG(tptr)) != NMPTRG(ENCLFUNCG(sptr)))
            continue;
          if (SCOPEG(sptr) && STYPEG(SCOPEG(sptr)) == ST_ALIAS &&
              SCOPEG(SCOPEG(sptr)) &&
              STYPEG(SCOPEG(SCOPEG(sptr))) == ST_MODULE) {
            if (!SCOPEG(tptr) || STYPEG(SCOPEG(tptr)) != ST_ALIAS ||
                !SCOPEG(SCOPEG(sptr)) ||
                STYPEG(SCOPEG(SCOPEG(sptr))) != ST_MODULE) {
              continue;
            }
            if (NMPTRG(SCOPEG(SCOPEG(tptr))) != NMPTRG(SCOPEG(SCOPEG(sptr))))
              continue;
          }
        }
        if (HCCSYMG(tptr))
          continue;
        /* tptr and sptr are two functions with the same name */
        FUNCLINEP(sptr, tptr);
        IGNOREP(sptr, 1);
        DPDSCP(sptr, 0); /* don't generate interface block */
        break;
      }
      break;
    default:;
    }
  }
} /* use_old_subprograms */

static void
erase_entry(int oldsymavl)
{
  int sptr;
  for (sptr = oldsymavl; sptr < stb.stg_avail; ++sptr) {
    switch (STYPEG(sptr)) {
    case ST_ENTRY:
      STYPEP(sptr, ST_PROC);
      IGNOREP(sptr, 1);
      break;
    default:;
    }
  }
} /* erase_entry */

extern void remove_alias(int std, int ast);

/*
 * temp workaround: return TRUE if there is an optional argument
 * to the subprogram, and the actual argument is missing.
 */
static LOGICAL
missing_optionals(int sptrEntry, int astCall)
{
  int argt, nargs, arg, astArg;
  int dpdsc, sptrDummy;
  argt = A_ARGSG(astCall);
  nargs = A_ARGCNTG(astCall);
  dpdsc = DPDSCG(sptrEntry);
  if (!dpdsc)
    return FALSE;
  for (arg = 0; arg < nargs; ++arg) {
    sptrDummy = aux.dpdsc_base[dpdsc + arg];
    astArg = ARGT_ARG(argt, arg);
    if (sptrDummy && OPTARGG(sptrDummy) && astArg == astb.ptr0) {
      return TRUE;
    }
  }
  return FALSE;
} /* missing_optionals */

/*
 * If CALL/FUNC ast residing in std has been extracted, inline the called
 * program unit, and continue inlining iLevels times.
 * Return TRUE if inlining performed.
 */
static LOGICAL
inline_func(int std, int ast, int iLevels, int level, int *psptrEntry)
{
  int sptrCall, sptrEntry;
  int sptrMod, sptrHost;
  char *sFunc;
  const char *sMod, *sHost, *currMod;
  FI *pfi;
  LE *ple;
  char sExtFile[MAX_FNAME_LEN];
  FILE *fd;
  int stdstart, stdfinal, sptrHighCopy;

#if DEBUG
  if (flg.x[22] && gbl.func_count != flg.x[22])
    return FALSE;
#endif
  assert(A_TYPEG(ast) == A_FUNC || A_TYPEG(ast) == A_CALL,
         "inline_func: bad type", ast, 4);
  assert(A_TYPEG(A_LOPG(ast)) == A_ID, "inline_func: bad call type", ast, 4);
  remove_alias(std, ast);
  sptrCall = A_SPTRG(A_LOPG(ast));
  sFunc = SYMNAME(sptrCall);
  if (!INTERNALG(sptrCall)) {
    sptrHost = sptrCall;
    sHost = ".";
  } else {
    sptrHost = SCOPEG(sptrCall);
    sHost = SYMNAME(sptrHost);
  }
  /* see if this is a module subprogram */
  currMod = NULL;
  sMod = ".";
  if (INMODULEG(sptrHost)) {
    for (sptrMod = sptrHost; sptrMod > NOSYM; sptrMod = SCOPEG(sptrMod)) {
      if (STYPEG(sptrMod) == ST_MODULE) {
        sMod = SYMNAME(sptrMod);
        currMod = sMod;
        break;
      }
      if (SCOPEG(sptrMod) == sptrMod) {
        break;
      }
    }
  }

  if (pfiStart) {
    /* Determine if sptrCall is a function specified for inlining. */
    for (pfi = pfiStart; pfi; pfi = pfi->pfiNext)
      if (!strcmp(pfi->sFunc, sFunc))
        break;
    if (!pfi)
      return FALSE;
  }

  ple = find_libentry(sMod, sHost, sFunc);
  if (!ple) {
    return FALSE;
  }

  if (!XBIT(13, 0x400) && missing_optionals(sptrCall, ast)) {
    ccff_info(
        MSGNEGINLINER, "INL044", gbl.findex, gbl.funcline,
        "%module%separator%function not inlined: missing OPTIONAL arguments",
        "module=%s", currMod ? SYMNAME(sptrMod) : "", "separator=%s",
        currMod ? "::" : "", "function=%s", SYMNAME(sptrCall), NULL);
    return FALSE;
  }

  sprintf(sExtFile, "%s/%s", sExtDir, ple->sModFile);
  fd = fopen(sExtFile, "r");
  if (!fd)
    error(275, 4, gbl.lineno, sExtFile, NULL);
  stdEnd = STD_PREV(0);
  sptrHighCopy = sptrHigh = stb.stg_avail;
  if (import_inline(fd, sExtFile)) {
    ccff_info(
        MSGNEGINLINER, "INL045", gbl.findex, gbl.funcline,
        "%module%separator%function not inlined: conflicting declarations",
        "module=%s", currMod ? SYMNAME(sptrMod) : "", "separator=%s",
        currMod ? "::" : "", "function=%s", SYMNAME(sptrCall), NULL);
    return FALSE;
  }

  sptrEntry = find_entry(A_SPTRG(A_LOPG(ast)));

  if (STD_PREV(0) == stdEnd) {
    ccff_info(MSGNEGINLINER, "INL047", gbl.findex, gbl.funcline,
              "%module%separator%function not inlined: empty subprogram",
              "module=%s", currMod ? SYMNAME(sptrMod) : "", "separator=%s",
              currMod ? "::" : "", "function=%s", SYMNAME(sptrCall), NULL);
    remove_inlined_symbols(sptrHigh);
    return FALSE;
  }
  if (aliased_args(sptrEntry, ast)) {
    ccff_info(MSGNEGINLINER, "INL048", gbl.findex, gbl.funcline,
              "%module%separator%function not inlined: argument aliasing",
              "module=%s", currMod ? SYMNAME(sptrMod) : "", "separator=%s",
              currMod ? "::" : "", "function=%s", SYMNAME(sptrCall), NULL);
    remove_inlined_stds();
    remove_inlined_symbols(sptrHigh);
    return FALSE;
  }

  use_old_subprograms(sptrHigh);

  modify_inlined_stds(sptrEntry, ast, STD_NEXT(stdEnd), STD_PREV(0));
  if (STD_PREV(0) == stdEnd) {
    ccff_info(MSGNEGINLINER, "INL049", gbl.findex, gbl.funcline,
              "%module%separator%function not inlined: no statements",
              "module=%s", currMod ? SYMNAME(sptrMod) : "", "separator=%s",
              currMod ? "::" : "", "function=%s", SYMNAME(sptrCall), NULL);
    remove_inlined_symbols(sptrHigh);
    return FALSE;
  }

#if DEBUG
  if (flg.x[18] != 0 && nskip < flg.x[18]) {
    ++nskip;
    ccff_info(MSGNEGINLINER, "INL050", gbl.findex, gbl.funcline,
              "%module%separator%function not inlined: skipped", "module=%s",
              currMod ? SYMNAME(sptrMod) : "", "separator=%s",
              currMod ? "::" : "", "function=%s", SYMNAME(sptrCall), NULL);
    remove_inlined_stds();
    remove_inlined_symbols(sptrHigh);
    return FALSE;
  }
#endif

  /* Insert the inlined STDs before std. */
  stdstart = STD_NEXT(stdEnd);
  stdfinal = STD_PREV(0);
  if (XBIT(49, 0x4000)) {
    /* line-level profiling */
    int std;
    for (std = stdstart; std; std = STD_NEXT(std)) {
      STD_LINENO(std) = 0;
      if (std == stdfinal)
        break;
    }
  }
  STD_NEXT(stdEnd) = 0;
  STD_PREV(0) = stdEnd;
  STD_PREV(stdstart) = STD_PREV(std);
  STD_NEXT(STD_PREV(std)) = stdstart;
  STD_NEXT(stdfinal) = std;
  STD_PREV(std) = stdfinal;

  /* change inlined ST_ENTRY */
  erase_entry(sptrHigh);

  if (XBIT(0, 8)) {
    ccff_info(
        MSGNEGINLINER, "INL051", gbl.findex, gbl.funcline,
        "%module%separator%function inlined", "module=%s",
        sptrHost == sptrCall && INMODULEG(sptrHost) ? SYMNAME(sptrMod) : "",
        "separator=%s", sptrHost == sptrCall && INMODULEG(sptrHost) ? "::" : "",
        "function=%s", SYMNAME(sptrCall), NULL);
  }
  ++ninlined;
  inline_stds(stdstart, stdfinal, iLevels - 1, level + 1);

  if (psptrEntry)
    *psptrEntry = sptrEntry;
  return TRUE;
}

/*
 * Modify inlined STDs between stdstart & stdlast for call AST ast.
 * Remove the END and RETURN statements.
 * Move labels on branching statements to CONTINUE statements.
 * If an argument is INTENT IN, or the actual argument is not an
 * identifier or subscript, copy the actual to the dummy parameter
 * at the beginning of the inlined code.
 * If an argument is INTENT INOUT/OUT and the actual argument is an identifer
 * or subscript, replace all occurrences of the dummy parameter with the
 * actual.
 */
static void
modify_inlined_stds(int sptrEntry, int ast, int stdstart, int stdlast)
{
  switch (A_TYPEG(ast)) {
  case A_CALL:
  case A_FUNC:
    break;
  default:
    interr("modify_inlined_stds: unrecognized CALL", stdstart, 4);
  }

  /* Copy expressions and arguments with INTENT IN to dummy parameters. */
  ast = copy_inargs(sptrEntry, ast, stdstart, stdlast);

  /* Remove RETURN & END statements. */
  remove_returns(stdstart, stdlast);

  /* Move labels on branching constructs. */
  move_labels(stdstart, stdlast);

  /* Replace dummy parameters with INTENT OUT or INOUT with identifier
   * arguments. */
  rewrite_inlined_args(ast, sptrEntry, stdstart, stdlast);

  /* Allocate adjustable arrays. */
  allocate_adjarrs(stdstart, stdlast);
}

/*
 * Return the symbol table pointer to the first ENTRY description associated
 * with the program unit whose symbol table pointer is sptrCall. The search
 * begins after sptrHigh.
 */
static int
find_entry(int sptrCall)
{
  int sptr;
  int nmptr = NMPTRG(sptrCall);

  for (sptr = sptrHigh; sptr < stb.stg_avail; sptr++)
    if (STYPEG(sptr) == ST_ENTRY && NMPTRG(sptr) == nmptr)
      break;
  assert(sptr < stb.stg_avail, "find_entry: ENTRY symtab record not found", 0, 4);
  return sptr;
}

/*
 * Copy arguments within astCall that have INTENT IN, or are expressions,
 * to new variables, which are allocated
 * prior to stdstart, and deallocated after stdlast. Replace the original
 * variables within call/func astCall with the new variables, and return
 * the new call AST.
 */
static int
copy_inargs(int sptrEntry, int astCall, int stdstart, int stdlast)
{
  int dscend = DPDSCG(sptrEntry) + PARAMCTG(sptrEntry);
  int dsc;
  int sptrCall, sptrDummy, sptrCopy, sptrBnd;
  int nargs, arg;
  int argtNew;
  int astArg, astCopy, ast, astl, astu, asts, vastSubs[7], astAlloc;
  int shd;
  int dtyp;
  int ndims, dim;
  ADSC *adCopy;

  assert(A_TYPEG(astCall) == A_CALL || A_TYPEG(astCall) == A_FUNC,
         "copy_inargs: call not found", astCall, 4);
  sptrCall = A_SPTRG(A_LOPG(astCall));
  if (PARAMCTG(sptrEntry) != A_ARGCNTG(astCall)) {
    error(271, 2, gbl.lineno, SYMNAME(sptrCall), NULL);
    remove_inlined_stds();
    return astCall;
  }

  /* Find an argument that needs to be copied. */
  nargs = A_ARGCNTG(astCall);
  for (arg = 0; arg < nargs; arg++)
    if (make_arg_copy(sptrEntry, astCall, arg))
      break;
  if (arg == nargs)
    return astCall; /* No copies needed. */

  argtNew = mk_argt(nargs); /* Create a new argument descriptor. */
  arg = 0;
  for (dsc = DPDSCG(sptrEntry); dsc < dscend; dsc++, arg++) {
    astArg = ARGT_ARG(A_ARGSG(astCall), arg);
    ARGT_ARG(argtNew, arg) = astArg;
    if (!make_arg_copy(sptrEntry, astCall, arg))
      continue;
    sptrDummy = aux.dpdsc_base[dsc];
    if (A_TYPEG(astArg) == A_SUBSCR && !A_SHAPEG(astArg) &&
        DTY(DTYPEG(sptrDummy)) == TY_ARRAY)
      astArg = promote_assumsz_arg(sptrDummy, astArg);
    shd = A_SHAPEG(astArg);
    if (shd) {
      dtyp = DDTG(A_DTYPEG(astArg));
      ndims = SHD_NDIM(shd);
      sptrCopy = sym_get_array(SYMNAME(sptrDummy), "inl", dtyp, ndims);
      adCopy = AD_DPTR(DTYPEG(sptrCopy));
      for (dim = 0; dim < ndims; dim++) {
        /* Assign 1 to a new lower bound temp. */
        sptrBnd = sym_get_scalar("l", "inl", DT_INT);
        astl = mk_id(sptrBnd);
        ast = mk_assn_stmt(astl, astb.i1, DT_INT);
        add_stmt_before(ast, stdstart);
        AD_LWBD(adCopy, dim) = AD_LWAST(adCopy, dim) = astl;

        /* Compute the size of dimension dim. */
        ast = mk_binop(OP_SUB, SHD_UPB(shd, dim), SHD_LWB(shd, dim), DT_INT);
        asts = SHD_STRIDE(shd, dim);
        if (!asts)
          asts = astb.i1;
        ast = mk_binop(OP_ADD, asts, ast, DT_INT);
        if (asts != astb.i1)
          ast = mk_binop(OP_DIV, ast, asts, DT_INT);

        /* Assign the computed size to a new upper bound temp. */
        sptrBnd = sym_get_scalar("u", "inl", DT_INT);
        astu = mk_id(sptrBnd);
        ast = mk_assn_stmt(astu, ast, DT_INT);
        add_stmt_before(ast, stdstart);
        AD_UPBD(adCopy, dim) = AD_UPAST(adCopy, dim) = astu;

        vastSubs[dim] = mk_triple(astl, astu, 0);
      }
      /* Create ALLOCATE & DEALLOCATE statements for the new array. */
      astCopy = mk_id(sptrCopy);
      ast = mk_subscr(astCopy, vastSubs, ndims, DTYPEG(sptrCopy));
      astAlloc = mk_stmt(A_ALLOC, 0);
      A_TKNP(astAlloc, TK_ALLOCATE);
      A_LOPP(astAlloc, 0);
      A_SRCP(astAlloc, ast);
      add_stmt_before(astAlloc, stdstart);
      astAlloc = mk_stmt(A_ALLOC, 0);
      A_TKNP(astAlloc, TK_DEALLOCATE);
      A_LOPP(astAlloc, 0);
      A_SRCP(astAlloc, astCopy);
      add_stmt_after(astAlloc, stdlast);
    } else {
      sptrCopy = sym_get_scalar(SYMNAME(sptrDummy), "inl", A_DTYPEG(astArg));
      astCopy = mk_id(sptrCopy);
    }
    /* Create a statement for copying the actual argument to the temp. */
    ast = mk_assn_stmt(astCopy, astArg, A_DTYPEG(astArg));
    add_stmt_before(ast, stdstart);
    ARGT_ARG(argtNew, arg) = astCopy;
  }
  /* Create a new CALL/FUNC AST. */
  ast = mk_func_node(A_TYPEG(astCall), A_LOPG(astCall), nargs, argtNew);
  A_SHAPEP(ast, A_SHAPEG(astCall));
  A_DTYPEP(ast, A_DTYPEG(astCall));
  return ast;
}

/*
 * Remove all inlined statements.
 */
static void
remove_inlined_stds()
{
  STD_PREV(0) = stdEnd;
  STD_NEXT(stdEnd) = 0;
}

static int
removelist(int listhead, int oldsymavl)
{
  int l;
  for (l = listhead; l >= oldsymavl; l = SYMLKG(l))
    ;
  return l;
} /* removelist */

static int
removeautolist(int listhead, int oldsymavl)
{
  int l;
  for (l = listhead; l >= oldsymavl; l = AUTOBJG(l))
    ;
  return l;
} /* removeautolist */

static void
remove_inlined_symbols(int oldsymavl)
{
  int sptr, hashval, len;
  char *np;
  INT V[4];
  for (sptr = stb.stg_avail - 1; sptr >= oldsymavl; --sptr) {
    hashval = -1;
    switch (STYPEG(sptr)) {
    case ST_CONST:
      switch (DTY(DTYPEG(sptr))) {
      case TY_BINT:
      case TY_SINT:
      case TY_INT:
      case TY_BLOG:
      case TY_SLOG:
      case TY_LOG:
      case TY_WORD:
      case TY_REAL:
      case TY_DWORD:
      case TY_DBLE:
      case TY_CMPLX:
      case TY_INT8:
      case TY_LOG8:
        V[0] = CONVAL1G(sptr);
        V[1] = CONVAL2G(sptr);
        hashval = HASH_CON(V);
        if (hashval < 0)
          hashval = -hashval;
        break;
      case TY_QUAD:
      case TY_DCMPLX:
      case TY_QCMPLX:
      case TY_HOLL:
      case TY_NCHAR:
        V[0] = CONVAL1G(0);
        V[1] = CONVAL2G(0);
        V[2] = CONVAL3G(0);
        V[3] = CONVAL4G(0);
        hashval = HASH_CON(V);
        if (hashval < 0)
          hashval = -hashval;
        break;
      case TY_CHAR:
        np = stb.n_base + CONVAL1G(sptr);
        HASH_ID(hashval, np, strlen(np));
        if (hashval < 0)
          hashval = -hashval;
        break;
      default:
        interr("remove_inlined_symbols: unknown constant datatype", 0, 4);
        return;
      }
      break;
    case ST_UNKNOWN:
    case ST_IDENT:
    case ST_LABEL:
    case ST_STAG:
    case ST_MEMBER:
    case ST_VAR:
    case ST_ARRAY:
    case ST_DESCRIPTOR:
    case ST_STRUCT:
    case ST_UNION:
    case ST_CMBLK:
    case ST_NML:
    case ST_ENTRY:
    case ST_PROC:
    case ST_STFUNC:
    case ST_PARAM:
    case ST_INTRIN:
    case ST_GENERIC:
    case ST_USERGENERIC:
    case ST_PD:
    case ST_PLIST:
    case ST_ARRDSC:
    case ST_ALIAS:
    case ST_MODULE:
    case ST_TYPEDEF:
    case ST_OPERATOR:
    case ST_MODPROC:
    case ST_CONSTRUCT:
    case ST_CRAY:
      np = SYMNAME(sptr);
      if (np) {
        len = strlen(np);
        HASH_ID(hashval, np, len);
        if (hashval < 0)
          hashval = -hashval;
      }
      break;
    default:;
    }
    if (hashval >= 0) {
      int s, ps;
      for (s = stb.hashtb[hashval], ps = 0; s; s = HASHLKG(s)) {
        if (s == sptr) {
          if (ps) {
            HASHLKP(ps, HASHLKG(s));
          } else {
            stb.hashtb[hashval] = HASHLKG(s);
          }
          break;
        }
        ps = s;
      }
    }
  }

  /* remove from gbl. lists */
  gbl.cmblks = removelist(gbl.cmblks, oldsymavl);
  gbl.externs = removelist(gbl.externs, oldsymavl);
  gbl.consts = removelist(gbl.consts, oldsymavl);
  gbl.entries = removelist(gbl.entries, oldsymavl);
  gbl.statics = removelist(gbl.statics, oldsymavl);
  gbl.locals = removelist(gbl.locals, oldsymavl);
  gbl.asgnlbls = removelist(gbl.asgnlbls, oldsymavl);
  gbl.autobj = removeautolist(gbl.autobj, oldsymavl);
  gbl.tp_adjarr = removeautolist(gbl.tp_adjarr, oldsymavl);
  gbl.p_adjarr = removelist(gbl.p_adjarr, oldsymavl);
  stb.stg_avail = oldsymavl;
} /* remove_inlined_symbols */

/*
 * Promote an actual assumed-size array argument to an array section, and
 * return the new AST.
 * Expand 'n' dimensions, where 'n' is the number of dimensions of
 * sptrDummy.  Also, if the expanded dimension is the last assumed-size
 * dimension, check that sptrDummy is also assumed-size
 */
static int
promote_assumsz_arg(int sptrDummy, int astArg)
{
  int asd;
  int nsubs, sub;
  int astl, astu, vastSubs[7], ast;
  int dtyp;
  int dtypeDummy, nsubsDummy, sptr, dtype;

  if (A_TYPEG(astArg) != A_SUBSCR)
    return astArg;
  if (A_SHAPEG(astArg))
    return astArg;
  dtypeDummy = DTYPEG(sptrDummy);
  if (DTY(dtypeDummy) != TY_ARRAY)
    return astArg;
  nsubsDummy = ADD_NUMDIM(dtypeDummy);
  asd = A_ASDG(astArg);
  nsubs = ASD_NDIM(asd);
  if (nsubsDummy > nsubs)
    return astArg;
  for (sub = 0; sub < nsubs; sub++)
    vastSubs[sub] = ASD_SUBS(asd, sub);
  sptr = memsym_of_ast(astArg);
  dtype = DTYPEG(sptr);
  for (sub = 0; sub < nsubsDummy; ++sub) {
    astl = ASD_SUBS(asd, sub);
    /* subscripts must be equal to lower bound, except last one */
    if (astl != ADD_LWAST(dtype, sub) && sub != nsubsDummy - 1)
      return astArg;
    astu = ADD_UPAST(dtype, sub);
    if (astu != 0) {
      vastSubs[sub] = mk_triple(astl, astu, 0);
    } else {
      /* had better be last dimension of actual */
      if (sub != nsubs - 1)
        return astArg;
      /* actual had better be assumed-size */
      if (!ADD_ASSUMSZ(dtype))
        return astArg;
      /* dummy had better be assumed-size */
      if (!ADD_ASSUMSZ(dtypeDummy))
        return astArg;
      vastSubs[sub] = mk_triple(astl, astu, 0);
    }
  }
  dtyp = get_array_dtype(nsubsDummy, A_DTYPEG(astArg));
  ast = mk_subscr(A_LOPG(astArg), vastSubs, nsubs, dtyp);
  return ast;
}

/*
 * Create ALLOCATE statements for all local adjustable arrays whose
 * symbol table entries occur after sptrHigh. ALLOCATE statements
 * are added before stdstart; DEALLOCATE statements are added after stdlast.
 */
static void
allocate_adjarrs(int stdstart, int stdlast)
{
  int sptr;

  for (sptr = sptrHigh; sptr < stb.stg_avail; sptr++)
    if (STYPEG(sptr) == ST_ARRAY && SCG(sptr) == SC_LOCAL && ADJARRG(sptr))
      allocate_array(sptr, stdstart, stdlast);
}

/*
 * Create an ALLOCATE statement before stdstart for array sptr, and a
 * DEALLOCATE statement for the array after stdlast.
 */
static void
allocate_array(int sptr, int stdstart, int stdlast)
{
  ADSC *ad;
  int sptrBnd;
  int vastSubs[7], astArr, ast, astAlloc, astl, astu;
  int std1 = STD_PREV(stdstart), std2 = stdlast;
  int ndims, dim;

  if (STYPEG(sptr) != ST_ARRAY)
    return;

  ad = AD_DPTR(DTYPEG(sptr));
  ndims = AD_NUMDIM(ad);
  for (dim = 0; dim < ndims; dim++) {
    if (AD_LWBD(ad, dim) && A_TYPEG(AD_LWAST(ad, dim)) == A_ID) {
      /* Explicitly assign bound expression to the lower bound. */
      ast = mk_assn_stmt(AD_LWAST(ad, dim), AD_LWBD(ad, dim), DT_INT);
      std1 = add_stmt_after(ast, std1);
    }

    /* Create a new array bound. */
    sptrBnd = sym_get_scalar("l", "inl", DT_INT);
    astl = mk_id(sptrBnd);
    ast = mk_assn_stmt(mk_id(sptrBnd), AD_LWAST(ad, dim), DT_INT);
    std1 = add_stmt_after(ast, std1);
    AD_LWAST(ad, dim) = astl;
    AD_LWBD(ad, dim) = astl;
    if (AD_UPBD(ad, dim) && A_TYPEG(AD_UPAST(ad, dim)) == A_ID) {
      /* Explicitly assign bound expression to the upper bound. */
      ast = mk_assn_stmt(AD_UPAST(ad, dim), AD_UPBD(ad, dim), DT_INT);
      std1 = add_stmt_after(ast, std1);
    }

    /* Create a new array bound. */
    sptrBnd = sym_get_scalar("u", "inl", DT_INT);
    astu = mk_id(sptrBnd);
    ast = mk_assn_stmt(mk_id(sptrBnd), AD_UPAST(ad, dim), DT_INT);
    std1 = add_stmt_after(ast, std1);
    AD_UPAST(ad, dim) = astu;
    AD_UPBD(ad, dim) = astu;
    AD_EXTNTAST(ad, dim) = mk_extent(astl, astu, dim);
    vastSubs[dim] = mk_triple(astl, astu, 0);
  }
  astArr = mk_id(sptr);
  ast = mk_subscr(astArr, vastSubs, ndims, DTYPEG(sptr));
  astAlloc = mk_stmt(A_ALLOC, 0);
  A_TKNP(astAlloc, TK_ALLOCATE);
  A_LOPP(astAlloc, 0);
  A_SRCP(astAlloc, ast);
  std1 = add_stmt_after(astAlloc, std1);
  astAlloc = mk_stmt(A_ALLOC, 0);
  A_TKNP(astAlloc, TK_DEALLOCATE);
  A_LOPP(astAlloc, 0);
  A_SRCP(astAlloc, astArr);
  std2 = add_stmt_after(astAlloc, std2);
  ALLOCP(sptr, TRUE);
  ADJARRP(sptr, FALSE);
  AD_DEFER(ad) = TRUE;
  AD_ADJARR(ad) = FALSE;
}

/*
 * Remove RETURN & END statements within STDs between stdstart & stdlast.
 * Set line #s on all STDs.
 */
static void
remove_returns(int stdstart, int stdlast)
{
  int std;
  int ast, astStmt, astLbl;
  int sptrEnd;
  LOGICAL bEndFound = FALSE;

  /* Create a new label for the END statement. */
  sptrEnd = getlab();
  astLbl = mk_label(sptrEnd);

  for (std = stdstart; std != STD_NEXT(stdlast); std = STD_NEXT(std)) {
    STD_LINENO(std) = gbl.lineno;
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_RETURN:
      /* Create a GOTO statement to the END-statement's new label. */
      astStmt = mk_stmt(A_GOTO, 0);
      A_L1P(astStmt, astLbl);
      STD_AST(std) = astStmt;
      A_STDP(astStmt, std);
      RFCNTI(sptrEnd);
      break;
    case A_END:
      /* Remove the END statement, and create a labeled CONTINUE. */
      astStmt = mk_stmt(A_CONTINUE, 0);
      STD_AST(std) = astStmt;
      A_STDP(astStmt, std);
      STD_LABEL(std) = sptrEnd;
      bEndFound = TRUE;
      break;
    }
  }
  assert(!RFCNTG(sptrEnd) || bEndFound, "remove_returns: no END found",
         stdstart, 4);
}

/*
 * Move labels on branching statements to new CONTINUE statements.
 */
static void
move_labels(int stdstart, int stdlast)
{
  int std, stdnew;
  int ast, astnew;
  int sptrLbl;

  for (std = stdstart; std != STD_NEXT(stdlast); std = STD_NEXT(std)) {
    sptrLbl = STD_LABEL(std);
    if (!sptrLbl)
      continue;
    STD_LABEL(std) = 0;
    if (!RFCNTG(sptrLbl))
      continue;
    ast = STD_AST(std);
    switch (A_TYPEG(ast)) {
    case A_IF:
    case A_IFTHEN:
    case A_ENDIF:
    case A_AIF:
    case A_GOTO:
    case A_CGOTO:
    case A_AGOTO:
    case A_ASNGOTO:
    case A_DO:
    case A_DOWHILE:
    case A_ENDDO:
    case A_CALL:
    case A_ICALL:
    case A_WHERE:
    case A_FORALL:
      astnew = mk_stmt(A_CONTINUE, 0);
      stdnew = add_stmt_before(astnew, std);
      STD_LABEL(stdnew) = sptrLbl;
      break;
    case A_ELSE:
    case A_ELSEIF:
    case A_ELSEWHERE:
    case A_ENDWHERE:
    case A_ENDFORALL:
    case A_ELSEFORALL:
      interr("move_labels: statement not handled", std, 4);
      FLANG_FALLTHROUGH;
    default:
      STD_LABEL(std) = sptrLbl;
      break;
    }
  }
}

/*
 * Within ast, replace every occurrence of a sub-expression containing
 * a dummy parameter with an expression derived from the corresponding
 * actual parameter.
 */
static int
replace_parms(int ast, int *extra_arg)
{
  int iar;
  int astLop, astNew, astSS, astSub, vastSubs[7], ast1;
  int sptr, dtypeOld, dtypeNew, memOld, memNew, astMemNew;
  int asd, asdSS;
  int ndims, dim;
  LOGICAL bChanged;

  switch (A_TYPEG(ast)) {
  case A_ID:
    sptr = A_SPTRG(ast);
    for (iar = 1; iar < iaravail; iar++)
      if (sptr == AR_SPTRDUMMY(iar))
        break;
    if (iar < iaravail) {
      astNew = replace_arg(ast, AR_ASTACT(iar));
      ast_replace(ast, astNew);
    } else {
      if (STYPEG(sptr) == ST_PROC) {
        if (!HCCSYMG(sptr) && IGNOREG(sptr) &&
            sptr >= use_old_subprogram_limits[0] &&
            sptr <= use_old_subprogram_limits[1]) {
          int tptr;
          tptr = FUNCLINEG(sptr);
          astNew = mk_id(tptr);
          ast_replace(ast, astNew);
        } else if (IGNOREG(sptr)) {
          /* reset IGNORE flag, we have inlined the call */
          IGNOREP(sptr, 0);
        }
      }
    }
    return TRUE;
  case A_MEM:
    astLop = A_PARENTG(ast);
    ast_traverse(astLop, replace_parms, NULL, NULL);
    astNew = ast_rewrite(astLop);
    if (astNew == astLop) {
      ast_replace(ast, ast);
    } else {
      /* we have to find the corresponding member name in any new datatype */
      dtypeOld = A_DTYPEG(astLop);
      dtypeNew = A_DTYPEG(astNew);
      if (DTY(dtypeOld) == TY_ARRAY)
        dtypeOld = DTY(dtypeOld + 1);
      if (DTY(dtypeNew) == TY_ARRAY)
        dtypeNew = DTY(dtypeNew + 1);
      if (dtypeOld == dtypeNew) {
        astMemNew = A_MEMG(ast);
        memNew = A_SPTRG(astMemNew);
      } else {
        sptr = A_SPTRG(A_MEMG(ast));
        if (DTY(dtypeOld) != DTY(dtypeNew) || DTY(dtypeOld) != TY_DERIVED) {
          interr("replace_parms: unmatched derived types", ast, 4);
        }
        for (memOld = DTY(dtypeOld + 1), memNew = DTY(dtypeNew + 1);
             memOld > NOSYM && memNew > NOSYM;
             memOld = SYMLKG(memOld), memNew = SYMLKG(memNew)) {
          if (memOld == sptr)
            break;
        }
        if (memOld <= NOSYM || memNew <= NOSYM ||
            ADDRESSG(memOld) != ADDRESSG(memNew)) {
          interr("replace_parms: unmatched derived type members", ast, 4);
        }
        astMemNew = mk_id(memNew);
      }
      astNew = mk_member(astNew, astMemNew, DTYPEG(memNew));
      ast_replace(ast, astNew);
    }
    return TRUE;

  case A_SUBSCR:
    asd = A_ASDG(ast);
    bChanged = FALSE;
    ndims = ASD_NDIM(asd);
    for (dim = 0; dim < ndims; dim++) {
      astSub = ASD_SUBS(asd, dim);
      ast_traverse(astSub, replace_parms, NULL, NULL);
      astNew = ast_rewrite(astSub);
      vastSubs[dim] = astNew;
      bChanged |= astNew != astSub;
    }
    astLop = A_LOPG(ast);
    ast_traverse(astLop, replace_parms, NULL, NULL);
    astNew = ast_rewrite(astLop);
    astSS = 0;
    if (A_TYPEG(astNew) == A_SUBSCR) {
      astSS = astNew;
      asdSS = A_ASDG(astSS);
      astNew = A_LOPG(astNew);
    }
    bChanged |= astNew != astLop;
    ast1 = ast;
    if (astSS && ASD_NDIM(asdSS) != ndims) {
      int nSubs[7], j;
      bChanged = 1;
      j = 0;
      ndims = ASD_NDIM(asdSS);
      for (dim = 0; dim < ndims; ++dim) {
        astSub = ASD_SUBS(asdSS, dim);
        if (A_SHAPEG(astSub) || A_TYPEG(astSub) == A_TRIPLE) {
          nSubs[dim] = vastSubs[j++];
        } else {
          nSubs[dim] = astSub;
        }
      }
      for (dim = 0; dim < ndims; ++dim)
        vastSubs[dim] = nSubs[dim];
    }
    if (bChanged)
      ast1 = mk_subscr(astNew, vastSubs, ndims, A_DTYPEG(ast));
    ast_replace(ast, ast1);
    return TRUE;
  case A_INTR:
    /* look for PRESENT() call */
    if (A_OPTYPEG(ast) == I_PRESENT) {
      /* 1st argument just be A_ID */
      int argt, arg;
      INT v[4];
      if (A_ARGCNTG(ast) != 1) {
        interr("replace_parms: PRESENT with wrong argument count",
               A_ARGCNTG(ast), 4);
      }
      argt = A_ARGSG(ast);
      arg = ARGT_ARG(argt, 0);
      if (A_TYPEG(arg) != A_ID) {
        interr("replace_parms: PRESENT with wrong argument type", A_TYPEG(ast),
               4);
      }
      sptr = A_SPTRG(arg);
      for (iar = 1; iar < iaravail; ++iar)
        if (sptr == AR_SPTRDUMMY(iar))
          break;
      if (iar >= iaravail) {
        interr("replace_parms: PRESENT with bad argument symbol", sptr, 4);
      }
      ast1 = AR_ASTACT(iar);
      if (ast1 == astb.ptr0) {
        /* missing optional, replace by .false. */
        v[0] = 0;
        v[1] = 0;
        v[2] = 0;
        v[3] = 0;
        sptr = getcon(v, DT_LOG4);
        astNew = mk_cnst(sptr);
        ast_replace(ast, astNew);
        return TRUE;
      } else if (A_TYPEG(ast1) == A_ID) {
        /* get the symbol; if it is a dummy, replace it */
        sptr = sym_of_ast(ast1);
        if (SCG(sptr) == SC_DUMMY) {
          /* use the dummy in the PRESENT call */
          ast_replace(arg, ast1);
        } else {
          /* present optional, replace by .true. */
          v[0] = -1;
          v[1] = -1;
          v[2] = 0;
          v[3] = 0;
          sptr = getcon(v, DT_LOG4);
          astNew = mk_cnst(sptr);
          ast_replace(ast, astNew);
          return TRUE;
        }
      } else {
        /* present optional, replace by .true. */
        v[0] = -1;
        v[1] = -1;
        v[2] = 0;
        v[3] = 0;
        sptr = getcon(v, DT_LOG4);
        astNew = mk_cnst(sptr);
        ast_replace(ast, astNew);
        return TRUE;
      }
    }
    break;
  }
  return FALSE;
}

/*
 * Rewrite all STDs between stdstart & stdlast so that arguments in
 * AST astCall replace dummy parameters in sptrEntry.
 */
static void
rewrite_inlined_args(int astCall, int sptrEntry, int stdstart, int stdlast)
{
  int iar;
  int dsc, dscend = DPDSCG(sptrEntry) + PARAMCTG(sptrEntry);
  int arg, argnum;
  int astArg, ast;
  int std;
  int sptrDummy, sptr, dtype;
  int ndims, dim;
  ADSC *ad;

  /* Initialize the argument replacement table. */
  parbase = NULL;
  iarsize = 100;
  NEW(parbase, AR, iarsize);
  iaravail = 1;

  arg = 0;
  argnum = 0;
  for (dsc = DPDSCG(sptrEntry); dsc < dscend; dsc++, arg++) {
    ++argnum;
    astArg = ARGT_ARG(A_ARGSG(astCall), arg);
    sptrDummy = aux.dpdsc_base[dsc];
    if (OPTARGG(sptrDummy) && astArg == astb.ptr0) {
      iar = iaravail++;
      NEED(iaravail, parbase, AR, iarsize, iarsize + 100);
      BZERO(&parbase[iar], AR, 1);
      AR_SPTRDUMMY(iar) = sptrDummy;
      AR_ASTACT(iar) = astArg;
    }
    if (A_TYPEG(astArg) != A_ID && A_TYPEG(astArg) != A_SUBSCR &&
        A_TYPEG(astArg) != A_MEM)
      continue;
    if (A_TYPEG(astArg) == A_SUBSCR && A_SHAPEG(astArg) &&
        DTY(DTYPEG(sptrDummy)) == TY_ARRAY && ASUMSZG(sptrDummy)) {
      remove_inlined_stds();
      return;
    }
    if (A_TYPEG(astArg) == A_SUBSCR && !A_SHAPEG(astArg) &&
        DTY(DTYPEG(sptrDummy)) == TY_ARRAY)
      astArg = promote_assumsz_arg(sptrDummy, astArg);
    if (!tkr_match_arg(DTYPEG(sptrDummy), A_DTYPEG(astArg))) {
      if (XBIT(0, 8)) {
        int modname;
        modname = SCOPEG(sptrEntry);
        if (modname && STYPEG(modname) == ST_ALIAS)
          modname = SCOPEG(modname);
        if (modname && STYPEG(modname) != ST_MODULE)
          modname = 0;
        ccff_info(MSGNEGINLINER, "INL052", gbl.findex, gbl.lineno,
                  "%module%separator%function not inlined: argument "
                  "%arg:%argname type mismatch",
                  "module=%s", modname ? SYMNAME(modname) : "", "separator=%s",
                  modname ? "::" : "", "function=%s", SYMNAME(sptrEntry),
                  "arg=%d", argnum, "argname=%s", SYMNAME(sptrDummy), NULL);
      }
      remove_inlined_stds();
      return;
    }
    if (DTY(DTYPEG(sptrDummy)) == TY_ARRAY)
      assign_bounds(sptrDummy, astArg, stdstart);

    iar = iaravail++;
    NEED(iaravail, parbase, AR, iarsize, iarsize + 100);
    BZERO(&parbase[iar], AR, 1);
    AR_SPTRDUMMY(iar) = sptrDummy;
    AR_ASTACT(iar) = astArg;
    if (SDSCG(sptrDummy)) {
      int sptrArgx;
      sptrArgx = memsym_of_ast(astArg);
      if (SDSCG(sptrArgx)) {
        iar = iaravail++;
        NEED(iaravail, parbase, AR, iarsize, iarsize + 100);
        BZERO(&parbase[iar], AR, 1);
        AR_SPTRDUMMY(iar) = SDSCG(sptrDummy);
        AR_ASTACT(iar) = check_member(astArg, mk_id(SDSCG(sptrArgx)));
      }
    }
    if (MIDNUMG(sptrDummy)) {
      int sptrArgx;
      sptrArgx = memsym_of_ast(astArg);
      if (MIDNUMG(sptrArgx)) {
        iar = iaravail++;
        NEED(iaravail, parbase, AR, iarsize, iarsize + 100);
        BZERO(&parbase[iar], AR, 1);
        AR_SPTRDUMMY(iar) = MIDNUMG(sptrDummy);
        AR_ASTACT(iar) = check_member(astArg, mk_id(MIDNUMG(sptrArgx)));
      }
    }
  }

  for (std = stdstart; std != STD_NEXT(stdlast); std = STD_NEXT(std)) {
    ast = STD_AST(std);

    /* Replace dummy parameters with actuals. */
    if (ast) {
      ast_visit(1, 1);
      ast_traverse(ast, replace_parms, NULL, NULL);
      ast = ast_rewrite(ast);
      ast_unvisit();
    }

    STD_AST(std) = ast;
    A_STDP(ast, std);
  }

  /* Rewrite dummy arrays appearing in bounds of adjustable arrays. */
  for (sptr = sptrHigh; sptr < stb.stg_avail; sptr++) {
    dtype = DTYPEG(sptr);
    if (STYPEG(sptr) == ST_ARRAY && SCG(sptr) == SC_LOCAL && ADJARRG(sptr)) {
      ad = AD_DPTR(dtype);
      ndims = AD_NUMDIM(ad);
      for (dim = 0; dim < ndims; dim++) {
        /* Replace dummy parameters with actuals in the lower bound. */
        ast = AD_LWBD(ad, dim);
        if (ast) {
          ast_visit(1, 1);
          ast_traverse(ast, replace_parms, NULL, NULL);
          ast = ast_rewrite(ast);
          ast_unvisit();
        }
        AD_LWBD(ad, dim) = ast;

        /* Replace dummy parameters with actuals in the upper bound. */
        ast = AD_UPBD(ad, dim);
        if (ast) {
          ast_visit(1, 1);
          ast_traverse(ast, replace_parms, NULL, NULL);
          ast = ast_rewrite(ast);
          ast_unvisit();
        }
        AD_UPBD(ad, dim) = ast;
      }
    }
    dtype = DDTG(dtype); /* base type */
    if ((DTY(dtype) == TY_CHAR || DTY(dtype) == TY_NCHAR) && ADJLENG(sptr)) {
      ast = DTY(dtype + 1);
      if (ast) {
        ast_visit(1, 1);
        ast_traverse(ast, replace_parms, NULL, NULL);
        ast = ast_rewrite(ast);
        ast_unvisit();
        DTY(dtype + 1) = ast;
      }
      if (ADJLENG(sptr)) {
        /* treat like we would an automatic object;
         * create a pointer variable */
        if (MIDNUMG(sptr) == 0) {
          int ptr;
          ptr = sym_get_ptr(sptr);
          if (SCG(sptr) == SC_STATIC) {
            SCP(ptr, SC_STATIC);
          } else {
            SCP(ptr, SC_LOCAL);
          }
          MIDNUMP(sptr, ptr);
          SCP(sptr, SC_BASED);
        }
      }
    }
  }

  FREE(parbase);
  parbase = NULL;
}

/*
 * Replace an occurrence of astDummy with a reference to astAct. Return
 * the reference to astAct.
 */
static int
replace_arg(int astDummy, int astAct)
{
  int sptrAct, sptrDummy;
  int rankAct, dimAct, dim, rankDummy;
  int ast, astl, astu, asts, astSub, ast1;
  int vastSubs[7];
  int shd, shdDummy;
  int dtyp, needsubscripts;
  ADSC *adDummy;

  shd = A_SHAPEG(astAct);
  if (!shd)
    return astAct;
  sptrAct = memsym_of_ast(astAct);
  rankAct = rank_of_sym(sptrAct);
  sptrDummy = memsym_of_ast(astDummy);
  assert(STYPEG(sptrDummy) == ST_ARRAY || STYPEG(sptrDummy) == ST_DESCRIPTOR,
         "replace_arg: formal not array", astDummy, 4);
  /*
   * if we pass a full array, and the dummy we're replacing is the full array,
   * we don't need any subscripts
   */
  needsubscripts = 0;
  for (ast = astAct; ast;) {
    switch (A_TYPEG(ast)) {
    case A_ID:
      ast = 0;
      break;
    case A_MEM:
      if (!A_SHAPEG(ast)) {
        ast = A_LOPG(ast);
      } else {
        ast = 0; /* leave loop */
                 /* we're using the shape of the whole member */
      }
      break;
    default:
      needsubscripts = 1;
      ast = 0; /* leave loop */
    }
  }
  if (needsubscripts == 0) {
    for (ast = astDummy; ast;) {
      switch (A_TYPEG(ast)) {
      case A_ID:
        ast = 0;
        break;
      case A_MEM:
        ast = A_LOPG(ast);
        break;
      default:
        needsubscripts = 1;
        ast = 0; /* leave loop */
      }
    }
  }
  if (!needsubscripts) {
    return astAct;
  }

  adDummy = AD_DPTR(DTYPEG(sptrDummy));
  rankDummy = AD_NUMDIM(adDummy);
  dim = -1;
  for (dimAct = 0; dimAct < rankAct; dimAct++) {
    astSub = get_subscr(astAct, dimAct);
    if (!A_SHAPEG(astSub) && A_TYPEG(astSub) != A_TRIPLE) {
      vastSubs[dimAct] = astSub;
      continue;
    }
    dim++;
    astSub = get_subscr(astDummy, dim);
    /* Convert each subscript x to s*(x-l1)+l2, where:
     *	l1 is the lower bound of the dummy array,
     *	s is the stride on the array section of the argument array,
     *	l2 is the lower bound on the array section of the argument.
     */
    /* Compute l2 - s*l1 once. */
    ast1 =
        mk_binop(OP_MUL, SHD_STRIDE(shd, dim), AD_LWAST(adDummy, dim), DT_INT);
    ast1 = mk_binop(OP_SUB, SHD_LWB(shd, dim), ast1, DT_INT);
    if (A_TYPEG(astSub) == A_TRIPLE) {
      astl = mk_binop(OP_MUL, SHD_STRIDE(shd, dim), A_LBDG(astSub), DT_INT);
      astl = mk_binop(OP_ADD, ast1, astl, DT_INT);
      astu = mk_binop(OP_MUL, SHD_STRIDE(shd, dim), A_UPBDG(astSub), DT_INT);
      astu = mk_binop(OP_ADD, ast1, astu, DT_INT);
      asts = A_STRIDEG(astSub);
      if (!asts)
        asts = astb.i1;
      asts = mk_binop(OP_MUL, SHD_STRIDE(shd, dim), asts, DT_INT);
      ast = mk_triple(astl, astu, asts);
    } else if (A_SHAPEG(astSub)) {
      ast = mk_convert(SHD_STRIDE(shd, dim), A_DTYPEG(astSub));
      ast = mk_binop(OP_MUL, ast, astSub, A_DTYPEG(astSub));
      ast1 = mk_convert(ast1, A_DTYPEG(astSub));
      ast = mk_binop(OP_ADD, ast1, ast, A_DTYPEG(astSub));
    } else {
      ast = mk_binop(OP_MUL, SHD_STRIDE(shd, dim), astSub, DT_INT);
      ast = mk_binop(OP_ADD, ast1, ast, DT_INT);
    }
    vastSubs[dimAct] = ast;
  }
  dtyp = DDTG(A_DTYPEG(astAct));
  shdDummy = A_SHAPEG(astDummy);
  if (shdDummy)
    dtyp = get_array_dtype(SHD_NDIM(shdDummy), dtyp);
  if (A_TYPEG(astAct) == A_SUBSCR)
    astAct = A_LOPG(astAct);
  ast = mk_subscr(astAct, vastSubs, rankAct, dtyp);
  return ast;
}

/*
 * Replace the array bounds in sptrDummy with the array bounds of the
 * corresponding actual array parameter, astAct. Create
 * assignments prior to std of actual array bounds to dummy bounds.
 */
static void
assign_bounds(int sptrDummy, int astAct, int std)
{
  int sptrAct;
  int dim, ndimsAct;
  int ast;
  ADSC *adDummy;
  int shd;

  sptrAct = memsym_of_ast(astAct);
  ndimsAct = rank_of_sym(sptrAct);

  /* Make assignments of actual array bounds to dummy bounds. */
  shd = A_SHAPEG(astAct);
  assert(shd, "assign_bounds: array has no shape", sptrDummy, 4);
  ndimsAct = SHD_NDIM(shd);
  adDummy = AD_DPTR(DTYPEG(sptrDummy));
  assert(ndimsAct == AD_NUMDIM(adDummy), "assign_bounds: different ranks",
         sptrDummy, 4);
  for (dim = 0; dim < ndimsAct; dim++) {
    if (A_TYPEG(AD_LWAST(adDummy, dim)) == A_ID) {
      /* dummylb = actuallb */
      ast = mk_assn_stmt(AD_LWAST(adDummy, dim), SHD_LWB(shd, dim), DT_INT);
      add_stmt_before(ast, std);
    }
    if (A_TYPEG(AD_UPAST(adDummy, dim)) == A_ID) {
      if (A_TYPEG(AD_LWAST(adDummy, dim)) == A_ID ||
          SHD_LWB(shd, dim) == AD_LWAST(adDummy, dim)) {
        /* dummyub = actualub */
        ast = mk_assn_stmt(AD_UPAST(adDummy, dim), SHD_UPB(shd, dim), DT_INT);
      } else {
        /* dummyub = actualub-(actuallb-dummylb) */
        ast =
            mk_binop(OP_SUB, SHD_LWB(shd, dim), AD_LWAST(adDummy, dim), DT_INT);
        ast = mk_binop(OP_SUB, SHD_UPB(shd, dim), ast, DT_INT);
        ast = mk_assn_stmt(AD_UPAST(adDummy, dim), ast, DT_INT);
      }
      add_stmt_before(ast, std);
    }
  }
}

/*
 * Return subscript #dim of array reference ast.
 */
static int
get_subscr(int ast, unsigned int dim)
{
  int dtyp;
  ADSC *ad;
  int astSub;
  int asd;

  switch (A_TYPEG(ast)) {
  case A_ID:
  case A_MEM:
    dtyp = A_DTYPEG(ast);
    assert(DTY(dtyp) == TY_ARRAY, "get_subscr: non-array reference", ast, 4);
    ad = AD_DPTR(dtyp);
    assert(dim < AD_NUMDIM(ad), "get_subscr: bad dim", ast, 4);
    astSub = mk_triple(AD_LWAST(ad, dim), AD_UPAST(ad, dim), astb.i1);
    return astSub;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    assert(dim < ASD_NDIM(asd), "get_subscr: bad subscript", ast, 4);
    astSub = ASD_SUBS(asd, dim);
    return astSub;
  default:
    interr("get_subscr: unknown type", ast, 4);
    return 0;
  }
}

static void
load_inline_file(char *sDir)
{
  char sHintsFile[MAX_FNAME_LEN];
  char sLine[MAX_LINE_LEN];
  int iStat;
  FILE *fd;

  /* Open Hints file. */
  iStat = fndpath("Inline", sHintsFile, MAX_FNAME_LEN, sDir);
  /* File found? */
  if (iStat)
    return;
  fd = fopen(sHintsFile, "r");
  if (!fd)
    return;

  /* read the hints file, record each routine in the file */
  while (fgets(sLine, MAX_LINE_LEN, fd)) {
    char *ch;
    for (ch = sLine; *ch; ++ch) {
      if (*ch == '\n') {
        *ch = '\0';
        break;
      }
    }
    store_funcname(sLine);
  }
  fclose(fd);
} /* load_inline_file */

/*
 * Save the extract directory obtained from a command line switch.
 */
static void
load_TOC(char *sDir)
{
  int iStat;
  char *sStat;
  FILE *fd;
  char sTOCFile[MAX_FNAME_LEN];
  char sLine[MAX_LINE_LEN];
  char sFunc[MAX_FNAME_LEN];
  char sHost[MAX_FNAME_LEN];
  char sMod[MAX_FNAME_LEN];
  char sModFile[MAX_FNAME_LEN];
  int iVer;
  LE *ple;

  sExtDir = sDir;
  pleStart = NULL;

  /* Open TOC file. */
  iStat = fndpath("TOC", sTOCFile, MAX_FNAME_LEN, sDir);
  if (iStat)
    /* TOC not found. */
    return;

  fd = fopen(sTOCFile, "r");
  if (!fd)
    ERROR("extractor/inliner: TOC file not found", 0);

  /* Read the TOC's header line. */
  sStat = fgets(sLine, MAX_LINE_LEN, fd);
  if (!sStat)
    return;

  iStat = sscanf(sLine, TOC_HEADER_FMT, &iVer);
  if (iStat != 1)
    ERROR("extractor/inliner: bad TOC file header", 0);
  if (iVer != INLINER_VERSION)
    ERROR("extractor/inliner: incorrect inliner version in TOC", 0);

  /* Read the TOC file and create LE records for each line. */
  while (fgets(sLine, MAX_LINE_LEN, fd)) {
    ple = (LE *)getitem(PERM_AREA, sizeof(LE));
    if (!ple)
      ERROR("extractor/inliner: can't allocate memory", 1);
    iStat = sscanf(sLine, TOC_ENTRY_FMT, sMod, sHost, sFunc, sModFile);
    if (iStat != 4)
      ERROR("extractor/inliner: bad TOC line", 0);
    ple->sMod = STASH(sMod, PERM_AREA);
    ple->sHost = STASH(sHost, PERM_AREA);
    ple->sFunc = STASH(sFunc, PERM_AREA);
    ple->sModFile = STASH(sModFile, PERM_AREA);
    ple->pleNext = pleStart;
    pleStart = ple;
  }
  fclose(fd);
  load_inline_file(sDir);
}

/*
 * Save a subroutine name that should be inlined.
 */
static void
store_funcname(char *sFunc)
{
  FI *pfi = (FI *)getitem(PERM_AREA, sizeof(FI));

  if (!pfi)
    ERROR("allocator: can't allocate memory", 2);
  pfi->sFunc = STASH(sFunc, PERM_AREA);
  pfi->pfiNext = pfiStart;
  pfiStart = pfi;
}

/*
 * Return a pointer to a library entry record associated with program
 * unit name sFunc. Return NULL if none found.
 */
static LE *
find_libentry(const char *sMod, const char *sHost, char *sFunc)
{
  LE *ple;

  for (ple = pleStart; ple; ple = ple->pleNext)
    if (!strcmp(ple->sFunc, sFunc) && !strcmp(ple->sHost, sHost) &&
        !strcmp(ple->sMod, sMod))
      return ple;
  return NULL;
}

/*
 * Return TRUE if the data type of a dummy argument, dtypDummy, has the
 * same type-kind-rank as the data type of an actual argument, dtypAct.
 */
static LOGICAL
tkr_match_arg(int dtypDummy, int dtypAct)
{
  int rankDummy, rankAct;

  /* Check scalar types. */
  if (!tk_match_arg(dtypDummy, dtypAct, FALSE))
    return FALSE;

  /* Check ranks. */
  rankDummy = rank_of(dtypDummy);
  rankAct = rank_of(dtypDummy);
  return (rankDummy == rankAct);
}

/*
 * Return TRUE if any arguments within CALL/FUNC AST astCall that are
 * valid left-hand side expressions occur in other expressions within the
 * call. If so, aliasing could occur.
 */
static LOGICAL
aliased_args(int sptrEntry, int astCall)
{
  int arg, arg1, nargs;
  int argt;
  int astArg, astArg1;

  assert(A_TYPEG(astCall) == A_CALL || A_TYPEG(astCall) == A_FUNC,
         "aliased_args: missing call", astCall, 4);
  assert(STYPEG(sptrEntry) == ST_ENTRY, "aliased_args: missing entry",
         sptrEntry, 4);
  argt = A_ARGSG(astCall);
  nargs = A_ARGCNTG(astCall);
  for (arg = 0; arg < nargs; arg++) {
    if (make_arg_copy(sptrEntry, astCall, arg))
      continue;
    astArg = ARGT_ARG(argt, arg);

    /* Check for occurrence of argAst in other arguments. */
    for (arg1 = 0; arg1 < nargs; arg1++) {
      if (arg == arg1)
        continue;
      if (make_arg_copy(sptrEntry, astCall, arg1))
        continue;
      astArg1 = ARGT_ARG(argt, arg1);
      if (contains_ast(astArg1, astArg))
        return TRUE;
    }
  }
  return FALSE;
}

/*
 * Return TRUE if argument #arg within astCall should be copied at entry
 * to the procedure whose entry symbol table pointer is sptrEntry.
 */
static LOGICAL
make_arg_copy(int sptrEntry, int astCall, unsigned int arg)
{
  int dsc;
  int sptrDummy;
  int astArg;

  assert(A_TYPEG(astCall) == A_CALL || A_TYPEG(astCall) == A_FUNC,
         "make_arg_copy: missing call", astCall, 4);
  assert(STYPEG(sptrEntry) == ST_ENTRY, "make_arg_copy: missing entry",
         sptrEntry, 4);
  assert(PARAMCTG(sptrEntry) > arg, "make_arg_copy: not enough dummy parms",
         sptrEntry, 4);
  assert(A_ARGCNTG(astCall) > arg, "make_arg_copy: not enough actual parms",
         astCall, 4);
  dsc = DPDSCG(sptrEntry) + arg;
  sptrDummy = aux.dpdsc_base[dsc];
  if (SCG(sptrDummy) == SC_EXTERN)
    return FALSE;
  assert(SCG(sptrDummy) == SC_DUMMY, "make_arg_copy: arg not dummy", sptrEntry,
         4);
  astArg = ARGT_ARG(A_ARGSG(astCall), arg);
  if (OPTARGG(sptrDummy) && astArg == astb.ptr0)
    return FALSE;
  return A_TYPEG(astArg) == A_SUBSTR || !A_ISLVAL(A_TYPEG(astArg));
}
