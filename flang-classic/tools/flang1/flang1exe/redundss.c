/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
    \brief Optimize redundant subscript computations.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "ast.h"
#include "optimize.h"
#include "rtlRtns.h"

#if DEBUG
#define Trace(a) TraceOutput a
/* print a message, continue */
#include <stdarg.h>
static void
TraceOutput(const char *fmt, ...)
{
  va_list argptr;
  va_start(argptr, fmt);
  if (DBGBIT(39, 1)) {
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
} /* TraceOutput */
#else
/* eliminate the trace output */
#define Trace(a)
#endif

/*
 * depth-first numbering of loops.
 * loops are numbered in Depth-first order, children numbered first
 *  +---- loop 1
 *  |+--- loop 2
 *  ||
 *  |+---
 *  |+--- loop 3
 *  ||+-- loop 4
 *  |||
 *  ||+--
 *  |+---
 *  +----
 * loops are ordered: 2,4,3,1
 * numbers are:
 *  loop  dfn  childdfn
 *    1    4    1
 *    2    1    1
 *    3    3    2
 *    4    2    2
 * The DFN is the loop's position in the loop ordering
 * ChildDFN is the DFN of the first descendant in the loop ordering
 * loop l1 is a descendant of loop l2 if dfn(l1) >= dfn(l1) and childdfn(l1) <= dfn(l2)
 */
 
 
typedef struct {
    int dfn, childdfn;
  }lpinfo_t;
STG_DECLARE(lpinfo, lpinfo_t);
#define LP_DFN(l)	lpinfo.stg_base[l].dfn
#define LP_CHILDDFN(l)	lpinfo.stg_base[l].childdfn

/* used more than once */
#define A_MOREG(s) astb.stg_base[s].f6
#define A_MOREP(s, v) (astb.stg_base[s].f6 = (v))
/*
 * clear A_OPT1, A_OPT2 fields
 * clear A_REPL, and the f6 flag
 */

static void
InitAST(void)
{
  int ast;
  for (ast = 0; ast < astb.stg_avail; ++ast) {
    A_OPT1P(ast, 0);
    A_OPT2P(ast, 0);
    A_REPLP(ast, 0);
    A_MOREP(ast, 0);
  }
} /* InitAST */

static int OnlyFG = 0;

/*
 * return '1' if ast 'ast' is marked as modified in loop 'l', or
 * in any loop contained inside of 'l'
 * use the DFS tree numbering saved in LP_DFN and LP_CHILDDFN
 */
static int
modified(int ast, int l)
{
  int m;
  m = A_OPT2G(ast); /* zero, or loop in which ast is modified */
  if (OnlyFG == 0) {
    int mdfs;
    mdfs = LP_DFN(m);
    if (mdfs <= LP_DFN(l) && mdfs >= LP_CHILDDFN(l)) {
      return 1;
    }
  } else {
    /* just compare numbers */
    if (m == l) {
      return 1;
    }
  }
  return 0;
} /* modified */

static int SSlisthead, SSlisttail;
STG_DECLARE(DT_Assigned,int);
#define DtypeAssigned(dt)	DT_Assigned.stg_base[dt]

static void
InitSSlist(void)
{
  SSlisthead = SSlisttail = 1;
} /* InitSSlist */

static void
clearSSlist(void)
{
  int ss, nextss;
  for (ss = SSlisthead; ss > 1; ss = nextss) {
    nextss = A_OPT1G(ss);
    A_OPT1P(ss, 0);
  }
  InitSSlist();
} /* clearSSlist */

#define SS_ALL 1
#define SS_MULTIPLE 2
#define SS_ROOT 4
#define SS_INV 8
#define SS_ALL_MULTIPLE 3
#define SS_INV_ROOT 12
#define SS_INV_ROOT_MULTIPLE 14

/*
 * findssexprs is called several times with a 'type' parameter
 * type == SS_ALL means put all subscript expressions and subexpressions
 *  on the SSlist.
 * type == SS_ROOT means put all 'root' subscript expressions on the SSlist,
 *  that is, all expressions which are directly used, or for which its
 *  parent is not an optimizable subscript subexpression.
 * type == SS_INV_ROOT means put all invariant 'root' subexpressions on the
 * SSlist
 * type == SS_ALL_MULTIPLE means the same as type == SS_ALL, but set A_MOREG if
 * it
 *  it put on the list more than once
 * type == SS_INV_ROOT_MULTIPLE means put all invariant 'root' subexpressions on
 * the
 *  SSlist that have at least 2 uses, found by A_MOREG != 0.
 */

static void
addSS(int ss, int type, int l)
{
  if (A_TYPEG(ss) == A_CONV )
    ss = A_LOPG(ss);
  switch (A_TYPEG(ss)) {
  case A_ID:
    if (!(type & SS_ALL))
      return;
    break;
  case A_CNST:
    return;
  }
  if (A_OPT1G(ss) != 0) {
    /* already on the list */
    if (type == SS_ALL_MULTIPLE) {
      A_MOREP(ss, 1);
    }
    return;
  }
#if DEBUG
  if (DBGBIT(39, 1)) {
    fprintf(gbl.dbgfil, "Adding subscript expression %d to list type %d for "
                        "loop/node %d, opt2=%d",
            ss, type, l, A_OPT2G(ss));
    if (DBGBIT(39, 4)) {
      fprintf(gbl.dbgfil, "  ");
      printast(ss);
    }
    fprintf(gbl.dbgfil, "\n");
  }
#endif
  if (SSlisttail == 1) {
    SSlisthead = ss;
  } else {
    A_OPT1P(SSlisttail, ss);
  }
  SSlisttail = ss;
  A_OPT1P(ss, 1);
  if (type == SS_ALL_MULTIPLE) {
    A_MOREP(ss, 0); /* first time on list */
  }
} /* addSS */

/*
 * puts subexpressions on SSlist
 * return '0' if there is a subexpression that should not be added to the SSlist
 * return '1 'otherwise
 */
static int
markss(int ast, int type, int l)
{
  int lop, rop, sptr, ssflags;

  switch (A_TYPEG(ast)) {
  default: /* only (), binop, unop, ID, const, MEM,
            * section descriptor references are optimizable */
    return 0;
  case A_PAREN:
  case A_UNOP:
  case A_CONV:
    lop = markss(A_LOPG(ast), type, l);
    if (!lop)
      return 0;
    if (lop) {
      if (type & SS_ALL)
        addSS(A_LOPG(ast), type, l);
      if (type != SS_INV_ROOT_MULTIPLE || A_MOREG(ast))
        return 1;
    }
    break;
  case A_BINOP:
    lop = markss(A_LOPG(ast), type, l);
    rop = markss(A_ROPG(ast), type, l);
    if (lop && rop && (type != SS_INV_ROOT_MULTIPLE || A_MOREG(ast))) {
      if (type & SS_ALL) {
        addSS(A_LOPG(ast), type, l);
        addSS(A_ROPG(ast), type, l);
      }
      return 1;
    }
    if (lop)
      addSS(A_LOPG(ast), type, l);
    if (rop)
      addSS(A_ROPG(ast), type, l);
    break;
  case A_ID:
    if (((type & SS_INV_ROOT) == SS_INV_ROOT) && modified(ast, l))
      return 0;
    if (type != SS_INV_ROOT_MULTIPLE || A_MOREG(ast))
      return 1;
    break;
  case A_SUBSCR:
    if (((type & SS_INV_ROOT) == SS_INV_ROOT) && modified(ast, l))
      return 0;
    if (type == SS_INV_ROOT_MULTIPLE && A_MOREG(ast) == 0)
      return 0;
    sptr = 0;
    if (!XBIT(70, 0x8000000)) {
      sptr = sptr_of_subscript(ast);
      if (!POINTERG(sptr))
        sptr = 0;
    }
    lop = markss(A_LOPG(ast), type, l);
    if (lop) {
      int asd, s, all, ss;
      asd = A_ASDG(ast);
      all = 1;
      ssflags = 0;
      for (s = 0; s < ASD_NDIM(asd); ++s) {
        ss = ASD_SUBS(asd, s);
        if (markss(ss, type, l)) {
          ssflags |= (1 << s);
          if (type == SS_ALL && !sptr) {
            addSS(ss, type, l);
          }
        } else {
          all = 0;
        }
      }
      if (!sptr) {
        if (all) {
          if (type & SS_ALL) {
            addSS(A_LOPG(ast), type, l);
          }
          return 1;
        } else if (ssflags) {
          for (s = 0; s < ASD_NDIM(asd); ++s) {
            ss = ASD_SUBS(asd, s);
            if (ssflags & (1 << s)) {
              addSS(ss, type, l);
            }
          }
        }
      }
    }
    /* ### probably needs some work for section descriptor members */
    break;
  case A_CNST:
    return 1; /* ok as subexpression */
  case A_MEM:
    /* allow these */
    if (((type & SS_INV_ROOT) == SS_INV_ROOT) && modified(ast, l))
      return 0;
    if (type == SS_INV_ROOT_MULTIPLE && A_MOREG(ast) == 0)
      return 0;
    lop = markss(A_PARENTG(ast), type, l);
    if (lop)
      return 1;
    break;
  }
  return 0;
} /* markss */

/*
 * at a root subexpression, recurse to the subexpressions
 * put eligible subexpressions on the SSlist
 */
static int
markrootss(int root, int type, int l)
{
  if (markss(root, type, l)) {
    int lop, rop;
    /* for a root subexpression, don't put on the list if
     * it is a simple unary op on a constant, or a binary
     * op where one (or both) operands are constants */
    switch (A_TYPEG(root)) {
    case A_UNOP:
      lop = A_LOPG(root);
      if (A_TYPEG(lop) == A_CNST)
        return 0;
      break;
    case A_BINOP:
      lop = A_LOPG(root);
      rop = A_ROPG(root);
      if (A_TYPEG(lop) == A_CNST) {
        if (A_TYPEG(rop) != A_CNST && type == SS_INV_ROOT_MULTIPLE) {
          addSS(rop, type, l);
        }
        return 0;
      } else if (A_TYPEG(rop) == A_CNST) {
        if (type == SS_INV_ROOT_MULTIPLE) {
          addSS(lop, type, l);
        }
        return 0;
      }
      break;
    }
    addSS(root, type, l);
    return 1;
  }
  return 0;
} /* markrootss */

static int findsstype, findssl;

static int subscrnest = 0;
static LOGICAL
count_nest(int ast, int *dummy)
{
  if (A_TYPEG(ast) == A_SUBSCR) {
    if (!XBIT(70, 0x8000000)) {
      int sptr;
      sptr = sptr_of_subscript(ast);
      if (!POINTERG(sptr))
        return FALSE;
    }
    ++subscrnest;
  }
  return FALSE;
} /* count_nest */

/*
 * at an AST, look for a subscript expression.
 * if we are in a subscript, add subexpressions to SSlist
 */
static void
findss(int ast, int *dummy)
{
  if (A_TYPEG(ast) == A_SUBSCR) {
    if (subscrnest == 1) {
      int asd, s;
      asd = A_ASDG(ast);
      for (s = 0; s < ASD_NDIM(asd); ++s) {
        int ss;
        ss = ASD_SUBS(asd, s);
        /* don't mark simple subscripts */
        switch (A_TYPEG(ss)) {
        case A_PAREN:
        case A_UNOP:
        case A_CONV:
        case A_BINOP:
        case A_SUBSCR:
        case A_MEM:
          markrootss(ss, findsstype, findssl);
        }
      }
    }
    --subscrnest;
  }
} /* findss */

/*
 * number any expressions in subscripts in a single fg node
 */
static void
findssexprsfg(int l, int fg, int newstmts, int type)
{
  int std;
  for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
    int ast;
    ast = STD_AST(std);
    Trace(("statement %d in node %d in loop %d type %d", std, fg, l, type));
    if (std >= newstmts && A_TYPEG(ast) == A_ASN) {
      if (!markrootss(A_SRCG(ast), type, l)) {
        ast_traverse(ast, count_nest, findss, NULL);
      }
    } else {
      ast_traverse(ast, count_nest, findss, NULL);
    }
    if (std == FG_STDLAST(fg))
      break; /* leave loop */
  }
} /* findssexprsfg */

#if DEBUG
static void
dumpSSlist(void)
{
  int ast;
  for (ast = SSlisthead; ast > 1; ast = A_OPT1G(ast)) {
    fprintf(gbl.dbgfil, "ast %d on list, OPT2=%d  ", ast, A_OPT2G(ast));

    printast(ast);
    fprintf(gbl.dbgfil, "\n");
  }
} /* dumpSSlist */
#endif

/*
 * find and number any expressions in subscripts in a single loop
 */
static void
findssexprs(int l, int newstmts, int type, int onlyfg, int parlevel)
{
  /* build a list starting at SSlisthead, linked by A_OPT1 fields,
   * terminated by '1', of optimizable expressions that appear in
   * subscripts in loops.  We terminate by '1' so we can easily tell
   * whether an ast is already on the list (A_OPT1(ast) != 0 means on
   * the list somewhere, A_OPT1(ast) == 0 means not on the list) */
  clearSSlist();
  Trace(("findssexprs, loop %d, type %d, parlevel %d", l, type, parlevel));
  ast_visit(1, 1);
  findsstype = type;
  findssl = l;
  if (onlyfg) {
    findssexprsfg(l, onlyfg, newstmts, type);
  } else {
    int fg;
    for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
      if (FG_PAR(fg) == parlevel)
        findssexprsfg(l, fg, newstmts, type);
    }
  }
  ast_unvisit();

#if DEBUG
  if (DBGBIT(39, 1))
    dumpSSlist();
#endif
} /* findssexprs */

/* has a CALL appeared in this loop? */
static int CallAppears = 0;

/*
 * mark ast 'ast' as modified in loop 'l'
 */
static void
mark(int ast, int l)
{
  int a, aa, sptr, dtype;
  a = ast;
  while (a > 0) {
    aa = a;
    switch (A_TYPEG(a)) {
    case A_ID:
      A_OPT2P(a, l);
      sptr = A_SPTRG(a);
      if (TARGETG(sptr) || F90POINTERG(sptr)) {
        dtype = DTYPEG(sptr);
        dtype = DDTG(dtype);
        DtypeAssigned(dtype) = l;
      }
      a = 0;
      break;
    case A_SUBSCR:
      A_OPT2P(a, l);
      a = A_LOPG(a);
      break;
    case A_SUBSTR:
      A_OPT2P(a, l);
      a = A_LOPG(a);
      break;
    case A_MEM:
      A_OPT2P(a, l);
      A_OPT2P(A_MEMG(a), l);
      sptr = A_SPTRG(A_MEMG(a));
      if (TARGETG(sptr) || F90POINTERG(sptr)) {
        dtype = DTYPEG(sptr);
        dtype = DDTG(dtype);
        DtypeAssigned(dtype) = l;
      }
      a = A_PARENTG(a);
      break;
    case A_CNST:
      aa = 0;
      a = 0;
      break;
    case A_BINOP:
      mark(A_LOPG(a), l);
      mark(A_ROPG(a), l);
      a = 0;
      aa = 0;
      break;
    default:
      interr("redundss:mark unexpected AST type", A_TYPEG(a), 3);
      a = 0;
      aa = 0;
      break;
    }
#if DEBUG
    if (DBGBIT(39, 2) && aa) {
      fprintf(gbl.dbgfil, "Modified ast %d = ", aa);
      printast(aa);
      fprintf(gbl.dbgfil, "\n");
    }
#endif
  }
} /* mark */

static void
markmodified(int ast, int *pl)
{
  int argt, argcnt, arg, a, astli, l, sptr, dtype;
  int subr, subrast, paramct, dpdsc, templatecall;
  l = *pl;
  switch (A_TYPEG(ast)) {
  /* for statement ASTs and function calls, mark modified variables */
  case A_DO:     /* modifies DO variable */
  case A_MP_PDO: /* modifies DO variable */
    mark(A_DOVARG(ast), l);
    break;
  case A_ASN: /* modifies LHS variable */
    mark(A_DESTG(ast), l);
    break;
  case A_FUNC: /* potentially modifies arguments */
  case A_CALL: /* potentially modifies arguments */
    argt = A_ARGSG(ast);
    argcnt = A_ARGCNTG(ast);
    subrast = A_LOPG(ast);
    if (A_TYPEG(subrast) == A_ID) {
      subr = A_SPTRG(subrast);
      paramct = PARAMCTG(subr);
      dpdsc = DPDSCG(subr);
    } else {
      subr = -1;
      paramct = -1;
      dpdsc = -1;
    }
    templatecall = 0;
    if (A_TYPEG(ast) == A_CALL && subr > 0 &&
        getF90TmplSectRtn(SYMNAME(subr))) {
      templatecall = 1;
    }
    for (a = 0; a < argcnt; ++a) {
      int asym;
      arg = ARGT_ARG(argt, a);
      /* don't count descriptor arguments as modified
       * if they need to be so marked, they will be at the array
       * argument to which they correspond */
      asym = -1;
      if (a < paramct) {
        asym = aux.dpdsc_base[dpdsc + a];
      }
      switch (A_TYPEG(arg)) {
      case A_ID:
      case A_MEM:
        sptr = memsym_of_ast(arg);
        if (!DESCARRAYG(sptr) || (templatecall && a == 0) ||
            (!templatecall && dpdsc == 0)) {
          /* regular symbol, or 1st argument to template call,
           * or call to a function without an interface */
          mark(arg, l);
        }
        if (asym > NOSYM && POINTERG(asym) && SDSCG(sptr)) {
          /* there is an explicit interface
           * the corresponding dummy is a pointer dummy
           * there is a section descriptor */
          mark(mk_id(SDSCG(sptr)), l);
        }
        break;
      case A_SUBSTR:
        mark(arg, l);
        break;
      case A_SUBSCR:
        sptr = memsym_of_ast(arg);
        if (!DESCARRAYG(sptr)) {
          mark(arg, l);
        }
        break;
      }
    }
    a = A_LOPG(ast);
    if (A_TYPEG(a) != A_ID || !PUREG(A_SPTRG(a))) {
      CallAppears = l;
    }
    break;
  case A_ICALL: /* potentially modifies first argument */
    argcnt = A_ARGCNTG(ast);
    argt = A_ARGSG(ast);
    arg = ARGT_ARG(argt, 0);
    switch (A_TYPEG(arg)) {
    case A_ID:
    case A_MEM:
      mark(arg, l);
    }
    switch (A_OPTYPEG(ast)) {
    case I_PTR2_ASSIGN:
      /* get datatype of first argument */
      dtype = A_DTYPEG(arg);
      DtypeAssigned(dtype) = l;
      /* modifies the section descriptor also */
      mark(ARGT_ARG(argt, 1), l);
      break;

    case I_NULLIFY:
    case I_PTR_COPYOUT:
      if (argcnt > 1) {
        /* modifies the section descriptor also */
        mark(ARGT_ARG(argt, 1), l);
      }
      break;

    case I_COPYOUT:
      /* modifies the section descriptor also */
      mark(ARGT_ARG(argt, 2), l);
      break;

    case I_COPYIN:
      /* modifies the section descriptor also */
      mark(ARGT_ARG(argt, 3), l);
      break;

    case I_PTR_COPYIN:
      /* modifies the section descriptor also */
      mark(ARGT_ARG(argt, 4), l);
      break;
    }
    break;
  case A_ALLOC: /* modifies allocated object */
  case A_REDIM: /* modifies redimensioned object */
    mark(A_SRCG(ast), l);
    sptr = sym_of_ast(A_SRCG(ast));
    if (MIDNUMG(sptr)) {
      mark(mk_id(MIDNUMG(sptr)), l);
    }
    if (PTROFFG(sptr)) {
      mark(mk_id(PTROFFG(sptr)), l);
    }
    if (SDSCG(sptr)) {
      mark(mk_id(SDSCG(sptr)), l);
    }
    break;
  case A_FORALL: /* modified FORALL indices */
    for (astli = A_LISTG(ast); astli; astli = ASTLI_NEXT(astli)) {
      mark(mk_id(ASTLI_SPTR(astli)), l);
    }
    break;
  case A_REALIGN:      /* modifies realigned object */
  case A_REDISTRIBUTE: /* modifies redistributed object */
    mark(A_LOPG(ast), l);
    sptr = sym_of_ast(A_SRCG(ast));
    if (MIDNUMG(sptr)) {
      mark(mk_id(MIDNUMG(sptr)), l);
    }
    if (PTROFFG(sptr)) {
      mark(mk_id(PTROFFG(sptr)), l);
    }
    if (SDSCG(sptr)) {
      mark(mk_id(SDSCG(sptr)), l);
    }
    break;
  }
} /* markmodified */

/* return '1' if loop 'm' appears in loop 'l' */
static int
inloop(int m, int l)
{
  if (OnlyFG == 0) {
    int mdfs;
    mdfs = LP_DFN(m);
    if (mdfs <= LP_DFN(l) && mdfs >= LP_CHILDDFN(l)) {
      return 1;
    }
  } else {
    if (m == l) {
      return 1;
    }
  }
  return 0;
} /* inloop */

/* mark modified subexpressions for fg node */
static void
markmodifiedexprsfg(int l, int fg)
{
  int std;
  for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
    ast_traverse(STD_AST(std), NULL, markmodified, &l);
    if (std == FG_STDLAST(fg))
      break;
  }
} /* markmodifiedexprsfg */

/*
 * at loop l, find which subexpressions are modified at this loop
 * level, and mark them modified by setting A_OPT2 to the loop number
 */
static void
markmodifiedexprs(int l, int onlyfg)
{
  /* go through the nodes in the loop and statements in the node */
  /* use ast_traverse to get all function references */
  ast_visit(1, 1);
  if (onlyfg) {
    markmodifiedexprsfg(l, onlyfg);
  } else {
    int fg;
    for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
      markmodifiedexprsfg(l, fg);
    }
  }
  ast_unvisit_norepl();
} /* markmodifiedexprs */

/*
 * now propagate 'modified' from subexpressions
 * because of the way the list of subexpressions is ordered,
 * all subsubexpressions are visited first;
 * we have to go through ALL subsubexpressions, even if they are not
 * used in loop l, because they might have been used in a contained loop
 * of loop l, and we have to mark A_OPT2 with 'l' even in that case.
 */
static void
propagatemodified(int l)
{
  int ast, dtype;
  for (ast = SSlisthead; ast > 1; ast = A_OPT1G(ast)) {
    if (!modified(ast, l)) {
      int sptr, asd, s, mem;
      switch (A_TYPEG(ast)) {
      case A_PAREN:
      case A_UNOP:
      case A_CONV:
        if (modified(A_LOPG(ast), l)) {
          A_OPT2P(ast, l);
        }
        break;
      case A_BINOP:
        if (modified(A_LOPG(ast), l) || modified(A_ROPG(ast), l)) {
          A_OPT2P(ast, l);
        }
        break;
      case A_ID:
        /* may be set by markmodified */
        sptr = A_SPTRG(ast);
        switch (SCG(sptr)) {
        case SC_CMBLK:
        case SC_STATIC:
          /* call in this loop? */
          if (inloop(CallAppears, l)) {
            A_OPT2P(ast, l);
          }
          break;
        default:;
        }
        /* if 'target' or 'pointer', and any assignment to pointer of that
         * datatype in the loop, or any call in the loop,
         * we must assume it might be modified secretly */
        if (TARGETG(sptr) || POINTERG(sptr)) {
          /* call in this loop? */
          if (inloop(CallAppears, l)) {
            A_OPT2P(ast, l);
          }
          /* assignment to pointer of that datatype? */
          dtype = DTYPEG(sptr);
          dtype = DDTG(dtype);
          if (inloop(DtypeAssigned(dtype), l)) {
            A_OPT2P(ast, l);
          }
        }
        if (DESCARRAYG(sptr)) {
          int ast2;
          ast2 = mk_id(sptr);
          if (ast2 != ast && modified(ast2, l)) {
            A_OPT2P(ast, l);
          }
        }
        break;
      case A_CNST:
        break;
      case A_SUBSCR:
        if (modified(A_LOPG(ast), l)) {
          A_OPT2P(ast, l);
        }
        asd = A_ASDG(ast);
        for (s = 0; s < ASD_NDIM(asd); ++s) {
          if (modified(ASD_SUBS(asd, s), l)) {
            A_OPT2P(ast, l);
          }
        }
        break;
      case A_MEM:
        if (modified(A_PARENTG(ast), l)) {
          A_OPT2P(ast, l);
        }
        mem = A_MEMG(ast);
        if (modified(mem, l)) {
          A_OPT2P(ast, l);
        }
        sptr = A_SPTRG(mem);
        if (DESCARRAYG(sptr)) {
          int ast2;
          ast2 = mk_id(sptr);
          if (ast2 != ast && modified(ast2, l)) {
            A_OPT2P(ast, l);
          }
        }
        /* if 'target' or 'pointer', and any assignment to pointer of that
         * datatype in the loop, or any call in the loop,
         * we must assume it might be modified secretly */
        if (TARGETG(sptr) || POINTERG(sptr)) {
          /* call in this loop? */
          if (inloop(CallAppears, l)) {
            A_OPT2P(ast, l);
          }
          /* assignment to pointer of that datatype? */
          if (inloop(DtypeAssigned(DTYPEG(sptr)), l)) {
            A_OPT2P(ast, l);
          }
        }
        break;
      default:
        A_OPT2P(ast, l);
        break;
      }
    }
  }
#if DEBUG
  if (DBGBIT(39, 1))
    dumpSSlist();
#endif
} /* propagatemodified */

/* lop (+/-) rop, where lop or rop (or both) might be zero */
static int
addcomponents(int lop, int optype, int rop, int l)
{
  int ast;
  if (rop == 0) {
    ast = lop;
  } else if (lop == 0) {
    if (optype == OP_ADD) {
      ast = rop;
    } else if (optype == OP_SUB) {
      if (A_TYPEG(rop) == A_UNOP && A_OPTYPEG(rop) == OP_SUB) {
        /* -(-rop) == rop */
        ast = A_LOPG(rop);
      } else if (A_TYPEG(rop) == A_BINOP && A_OPTYPEG(rop) == OP_SUB) {
        /* -(rlop-rrop) = rrop-rlop */
        ast = mk_binop(optype, A_ROPG(rop), A_LOPG(rop), A_DTYPEG(rop));
        A_OPT2P(ast, A_OPT2G(rop));
      } else {
        ast = mk_unop(optype, rop, A_DTYPEG(rop));
        A_OPT2P(ast, A_OPT2G(rop));
      }
    } else {
      ast = mk_unop(optype, rop, A_DTYPEG(rop));
      A_OPT2P(ast, A_OPT2G(rop));
    }
  } else if (optype == OP_ADD) {
    /* look for (-a)+(-b), (-a)+b, a+(-b), (-a)+a, a+(-a) */
    if (A_TYPEG(lop) == A_UNOP && A_OPTYPEG(lop) == OP_SUB) {
      if (A_TYPEG(rop) == A_UNOP && A_OPTYPEG(rop) == OP_SUB) {
        int tmp;
        /* (-a)+(-b) turns into -(a+b) */
        lop = A_LOPG(lop);
        rop = A_LOPG(rop);
        tmp = mk_binop(OP_ADD, lop, rop, A_DTYPEG(rop));
        if (!modified(tmp, l)) {
          if (modified(lop, l) || modified(rop, l)) {
            A_OPT2P(tmp, l);
          }
        }
        ast = mk_unop(OP_SUB, tmp, A_DTYPEG(tmp));
        A_OPT2P(ast, A_OPT2G(tmp));
      } else {
        /* (-a) + b turns into b-a */
        lop = A_LOPG(lop);
        if (lop == rop) {
          /* -a + a */
          ast = 0;
        } else {
          ast = mk_binop(OP_SUB, rop, lop, A_DTYPEG(rop));
          if (!modified(ast, l)) {
            if (modified(lop, l) || modified(rop, l)) {
              A_OPT2P(ast, l);
            }
          }
        }
      }
    } else if (A_TYPEG(rop) == A_UNOP && A_OPTYPEG(rop) == OP_SUB) {
      /* a + (-b) turns into a-b */
      rop = A_LOPG(rop);
      if (lop == rop) {
        /* a - a */
        ast = 0;
      } else {
        ast = mk_binop(OP_SUB, lop, rop, A_DTYPEG(rop));
        if (!modified(ast, l)) {
          if (modified(lop, l) || modified(rop, l)) {
            A_OPT2P(ast, l);
          }
        }
      }
    } else {
      ast = mk_binop(OP_ADD, lop, rop, A_DTYPEG(rop));
      if (!modified(ast, l)) {
        if (modified(lop, l) || modified(rop, l)) {
          A_OPT2P(ast, l);
        }
      }
    }
  } else {
    /* look for (-a)-(-b), (-a)-b, a-(-b), a-a */
    if (A_TYPEG(lop) == A_UNOP && A_OPTYPEG(lop) == OP_SUB) {
      if (A_TYPEG(rop) == A_UNOP && A_OPTYPEG(rop) == OP_SUB) {
        /* (-a)-(-b) turns into b-a */
        lop = A_LOPG(lop);
        rop = A_LOPG(rop);
        if (lop == rop) {
          /* (-a) - (-a) */
          ast = 0;
        } else {
          ast = mk_binop(OP_SUB, rop, lop, A_DTYPEG(rop));
          if (!modified(ast, l)) {
            if (modified(lop, l) || modified(rop, l)) {
              A_OPT2P(ast, l);
            }
          }
        }
      } else {
        /* (-a) - b turns into -(a+b) */
        int tmp;
        lop = A_LOPG(lop);
        tmp = mk_binop(OP_ADD, rop, lop, A_DTYPEG(rop));
        if (!modified(tmp, l)) {
          if (modified(lop, l) || modified(rop, l)) {
            A_OPT2P(tmp, l);
          }
        }
        ast = mk_unop(OP_SUB, tmp, A_DTYPEG(tmp));
        A_OPT2P(ast, A_OPT2G(tmp));
      }
    } else if (A_TYPEG(rop) == A_UNOP && A_OPTYPEG(rop) == OP_SUB) {
      /* a - (-b) turns into a+b */
      rop = A_LOPG(rop);
      ast = mk_binop(OP_ADD, lop, rop, A_DTYPEG(rop));
      if (!modified(ast, l)) {
        if (modified(lop, l) || modified(rop, l)) {
          A_OPT2P(ast, l);
        }
      }
    } else {
      if (lop == rop) {
        /* a - a */
        ast = 0;
      } else {
        ast = mk_binop(OP_SUB, lop, rop, A_DTYPEG(rop));
        if (!modified(ast, l)) {
          if (modified(lop, l) || modified(rop, l)) {
            A_OPT2P(ast, l);
          }
        }
      }
    }
  }
  return ast;
} /* addcomponents */

/* -lop, look for -(-lop), -(l-r) */
static int
negatecomponent(int lop, int l)
{
  int ast;
  if (lop == 0) {
    ast = 0;
  } else if (A_TYPEG(lop) == A_UNOP && A_OPTYPEG(lop) == OP_SUB) {
    /* -(-lop) == lop */
    ast = A_LOPG(lop);
  } else if (A_TYPEG(lop) == A_BINOP && A_OPTYPEG(lop) == OP_SUB) {
    /* -(llop-lrop) = lrop-llop */
    ast = mk_binop(OP_SUB, A_ROPG(lop), A_LOPG(lop), A_DTYPEG(lop));
    A_OPT2P(ast, A_OPT2G(lop));
  } else {
    ast = mk_unop(OP_SUB, lop, A_DTYPEG(lop));
    A_OPT2P(ast, A_OPT2G(lop));
  }
  return ast;
} /* negatecomponent */

/*
 * lop*rop, where lop or rop (or both) might be zero
 * loop for (-lop)*(-rop), (-lop)*rop, eliminate negatives
 */
static int
mulcomponents(int lop, int rop, int l)
{
  int ast, negate;
  if (lop == 0 || rop == 0) {
    ast = 0;
  } else {
    negate = 1;
    if (A_TYPEG(lop) == A_UNOP && A_OPTYPEG(lop) == OP_SUB) {
      negate = -negate;
      lop = A_LOPG(lop);
    }
    if (A_TYPEG(rop) == A_UNOP && A_OPTYPEG(rop) == OP_SUB) {
      negate = -negate;
      rop = A_LOPG(rop);
    }
    if (lop == astb.i1) {
      ast = rop;
    } else if (rop == astb.i1) {
      ast = lop;
    } else {
      ast = mk_binop(OP_MUL, lop, rop, A_DTYPEG(rop));
      if (!modified(ast, l)) {
        if (modified(lop, l) || modified(rop, l)) {
          A_OPT2P(ast, l);
        }
      }
    }
    if (negate == -1) {
      int nast;
      nast = mk_unop(OP_SUB, ast, A_DTYPEG(ast));
      if (!modified(nast, l)) {
        if (modified(ast, l)) {
          A_OPT2P(nast, l);
        }
      }
      ast = nast;
    }
  }
  return ast;
} /* mulcomponents */

/*
 * given an ast, see if it is a product, and if so, build two trees,
 * one the product of components that are invariant in loop l, and one
 * the product of components that are variant in loop l
 */
static void
findmulcomponents(int ast, int l, int *pinv, int *pvar)
{
  int linv, lvar, rinv, rvar;
  if (ast == 0) {
    *pvar = 0;
    *pinv = 0;
    return;
  }
  /* default result, unless some other case applies */
  if (modified(ast, l) || A_TYPEG(ast) == A_CNST) {
    *pvar = ast;
    *pinv = astb.i1;
  } else {
    *pvar = astb.i1;
    *pinv = ast;
  }
  switch (A_TYPEG(ast)) {
  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      findmulcomponents(A_LOPG(ast), l, pinv, pvar);
      break;
    case OP_SUB:
      findmulcomponents(A_LOPG(ast), l, &linv, &lvar);
      /* negate one of them */
      *pinv = negatecomponent(linv, l);
      *pvar = lvar;
      break;
    }
    break;
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_MUL:
      findmulcomponents(A_LOPG(ast), l, &linv, &lvar);
      findmulcomponents(A_ROPG(ast), l, &rinv, &rvar);
      *pinv = mulcomponents(linv, rinv, l);
      *pvar = mulcomponents(lvar, rvar, l);
      break;
    }
    break;
  case A_PAREN:
    findmulcomponents(A_LOPG(ast), l, pinv, pvar);
    break;
  }
} /* findmulcomponents */

/*
 * given an ast, see if it is a sum, and if so, build two trees,
 * one the sum of components that are invariant in loop l, and one
 * the sum of components that are variant in loop l
 */
static void
findcomponents(int ast, int l, int *pinv, int *pvar)
{
  int linv, lvar, rinv, rvar, optype;
  /* default result, unless some other case applies */
  if (modified(ast, l) || A_TYPEG(ast) == A_CNST) {
    *pvar = ast;
    *pinv = 0;
  } else {
    *pvar = 0;
    *pinv = ast;
  }
  switch (A_TYPEG(ast)) {
  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      findcomponents(A_LOPG(ast), l, pinv, pvar);
      break;
    case OP_SUB:
      findcomponents(A_LOPG(ast), l, &linv, &lvar);
      /* negate both of them */
      *pinv = negatecomponent(linv, l);
      *pvar = negatecomponent(lvar, l);
      break;
    }
    break;
  case A_BINOP:
    optype = A_OPTYPEG(ast);
    switch (optype) {
    case OP_ADD:
    case OP_SUB:
      findcomponents(A_LOPG(ast), l, &linv, &lvar);
      findcomponents(A_ROPG(ast), l, &rinv, &rvar);
      *pinv = addcomponents(linv, optype, rinv, l);
      *pvar = addcomponents(lvar, optype, rvar, l);
      break;
    case OP_MUL:
      findcomponents(A_LOPG(ast), l, &linv, &lvar);
      findcomponents(A_ROPG(ast), l, &rinv, &rvar);
      if (lvar == 0) {
        int rvarinv, rvarvar;
        /* linv*(rinv+(rvarinv*rvarvar))  becomes
         * linv*rinv + (linv*rvarinv)*rvarvar */
        *pinv = mulcomponents(linv, rinv, l);
        findmulcomponents(rvar, l, &rvarinv, &rvarvar);
        linv = mulcomponents(linv, rvarinv, l);
        *pvar = mulcomponents(linv, rvarvar, l);
      } else if (rvar == 0) {
        int lvarinv, lvarvar;
        /* (linv+(lvarinv*lvarvar))*rinv  becomes
         * linv*rinv + (lvarinv*rinv)*lvarvar */
        *pinv = mulcomponents(linv, rinv, l);
        findmulcomponents(lvar, l, &lvarinv, &lvarvar);
        rinv = mulcomponents(lvarinv, rinv, l);
        *pvar = mulcomponents(lvarvar, rinv, l);
      } else {
        /* (lvar+linv)*(rinv+rvar)  just reassociate */
        *pinv = 0;
        lvar = addcomponents(lvar, OP_ADD, linv, l);
        rvar = addcomponents(rvar, OP_ADD, rinv, l);
        *pvar = mulcomponents(lvar, rvar, l);
      }
      break;
    }
    break;
  case A_PAREN:
    findcomponents(A_LOPG(ast), l, pinv, pvar);
    break;
  }
} /* findcomponents */

/*
 * return TRUE if ast is a sum of variant and invariant parts */
static LOGICAL
mixedsum(int ast, int l)
{
  LOGICAL mixed;
  int lop, rop, lmod, rmod;
  mixed = FALSE;
  switch (A_TYPEG(ast)) {
  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
      mixed = mixedsum(A_LOPG(ast), l);
    }
    break;
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
      lop = A_LOPG(ast);
      rop = A_ROPG(ast);
      lmod = modified(lop, l);
      rmod = modified(rop, l);
      if ((lmod && !rmod) || (!lmod && rmod)) {
        mixed = TRUE;
      } else if (mixedsum(lop, l) || mixedsum(rop, l)) {
        mixed = TRUE;
      }
      break;
    }
    break;
  case A_PAREN:
    mixed = mixedsum(A_LOPG(ast), l);
    break;
  }
  return mixed;
} /* mixedsum */

/*
 * given 'factor' that is invariant in loop 'l',
 * and 'sum' that is partially invariant in loop 'l',
 * distribute 'factor' over 'sum':
 *  (factor)*((sumvarinv*sumvarvar)+suminv) becomes
 *  ((factor)*(sumvarinv))*(sumvarvar) + (factor)*(suminv)
 */
static int
distribute(int factor, int sum, int l)
{
  int suminv, sumvar, sumvarinv, sumvarvar, ast;
  findcomponents(sum, l, &suminv, &sumvar);
  suminv = mulcomponents(factor, suminv, l);
  findmulcomponents(sumvar, l, &sumvarinv, &sumvarvar);
  factor = mulcomponents(factor, sumvarinv, l);
  sumvar = mulcomponents(factor, sumvarvar, l);
  ast = addcomponents(suminv, OP_ADD, sumvar, l);
  return ast;
} /* distribute */

/*
 * return a replacement ast that is reassociated and distributed
 * to give better code floating.
 * (i*k+1)*2+3 will become (i*(k*2))+((1*2)+3)
 * We compute a sum of a loop-invariant expression and a loop-variant one.
 * Leave the constant part in the 'loop-variant' part.
 */
static int
MakeReplacement(int ast, int l)
{
  int optype, rast = 0;
  int lop, linv, lvar, rop, rinv, rvar, subscr[MAXSUBS], numdim, asd, s, any;
  switch (A_TYPEG(ast)) {
  case A_PAREN:
    return MakeReplacement(A_LOPG(ast), l);
  case A_CONV:
    rast = MakeReplacement(A_LOPG(ast), l);
    if (rast == A_LOPG(ast)) {
      rast = ast;
    } else {
      rast = mk_convert(rast, A_DTYPEG(ast));
    }
    break;
  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_SUB:
      rast = MakeReplacement(A_LOPG(ast), l);
      rast = negatecomponent(rast, l);
      break;
    case OP_ADD:
      rast = MakeReplacement(A_LOPG(ast), l);
      break;
    default:
      rast = ast;
      break;
    }
    break;
  case A_BINOP:
    optype = A_OPTYPEG(ast);
    switch (optype) {
    case OP_ADD:
    case OP_SUB:
      /* find 'components' of lop and rop, linv, lvar, rinv, rvar.
       * add (linv+rinv)+(lvar+rvar) */
      findcomponents(A_LOPG(ast), l, &linv, &lvar);
      findcomponents(A_ROPG(ast), l, &rinv, &rvar);
      linv = addcomponents(linv, optype, rinv, l);
      lvar = addcomponents(lvar, optype, rvar, l);
      rast = addcomponents(linv, OP_ADD, lvar, l);
      break;
    case OP_MUL:
      /* multiply is the most difficult case
       * ((a+b)*(c+d)) * ((e+f)*(g+h))
       * If one side is invariant and the other is variant ADD or SUB,
       * but not completely variant, distribute the one side over the
       * other.
       * If either or both sides are multiplies, find multiplicative
       * components and reassociate as for sums */
      lop = MakeReplacement(A_LOPG(ast), l);
      rop = MakeReplacement(A_ROPG(ast), l);
      rast = ast;
      if (!modified(lop, l) && mixedsum(rop, l)) {
        rast = distribute(lop, rop, l);
      } else if (!modified(rop, l) && mixedsum(lop, l)) {
        rast = distribute(rop, lop, l);
      } else {
        findmulcomponents(ast, l, &linv, &lvar);
        rast = mulcomponents(linv, lvar, l);
      }
      break;
    default:
      rast = ast;
      break;
    }
    break;
  case A_SUBSCR:
    /* see if any subscripts can be modified */
    if (!XBIT(70, 0x8000000)) {
      int sptr;
      sptr = sptr_of_subscript(ast);
      if (!POINTERG(sptr))
        break;
    }
    rast = ast;
    asd = A_ASDG(ast);
    numdim = ASD_NDIM(asd);
    any = 0;
    for (s = 0; s < numdim; ++s) {
      subscr[s] = MakeReplacement(ASD_SUBS(asd, s), l);
      if (subscr[s] != ASD_SUBS(asd, s))
        ++any;
    }
    if (any) {
      rast = mk_subscr(A_LOPG(ast), subscr, numdim, A_DTYPEG(ast));
    }
    break;
  default:
    rast = ast;
  }
  return rast;
} /* MakeReplacement */

/* count how many ast_replace calls are made */
static int gany = 0;

/*
 * make an assignment for each expression on SSlisthead
 */
static int
addssassignments(int before, int beforefg, int rewrite, int par, int task)
{
  int ast, any;
  any = 0;
  for (ast = SSlisthead; ast > 1; ast = A_OPT1G(ast)) {
    /* ast is used in this loop */
    int lhs, asn, std, rhs;
    int dt;
    ++gany;
    ++any;
#if DEBUG
    if (DBGBIT(39, 4)) {
      Trace(("Remove ast %d, OPT2=%d, gany=%d", ast, A_OPT2G(ast), gany));
      printast(ast);
      Trace((""));
    }
#endif
    dt = A_DTYPEG(ast);
    if (!par && !task) {
      lhs = mk_id(getcctmp_sc('r', ast, ST_VAR, dt, SC_LOCAL));
    } else {
      lhs = mk_id(getcctmp_sc('r', ast, ST_VAR, dt, SC_PRIVATE));
    }

    asn = mk_stmt(A_ASN, A_DTYPEG(lhs));
    rhs = ast_rewrite(ast);
    A_SRCP(asn, rhs);
    A_DESTP(asn, lhs);
    std = add_stmt_before(asn, before);
    STD_FG(std) = beforefg;
    STD_PAR(std) = par;
    STD_TASK(std) = task;
    Trace(("add statement %d in node %d", std, beforefg));
    ast_replace(ast, lhs);
  }
  return any;
} /* addssassignments */

static void
rewritefg(int fg)
{
  int std;
  for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
    int ast, laststd = 0;
    if (std == FG_STDLAST(fg))
      laststd = 1;
    ast = STD_AST(std);
    ast = ast_rewrite(ast);
    STD_AST(std) = ast;
    if (A_TYPEG(ast) == A_ASN && A_DESTG(ast) == A_SRCG(ast)) {
      Trace(("delete useless assignment %d", std));
      if (STD_LABEL(std) == 0) {
        /* flow around it */
        STD_DELETE(std) = 1;
        STD_NEXT(STD_PREV(std)) = STD_NEXT(std);
        STD_PREV(STD_NEXT(std)) = STD_PREV(std);
        /* remove from flow graph */
        if (std == FG_STDFIRST(fg) && std == FG_STDLAST(fg)) {
          FG_STDFIRST(fg) = 0;
          FG_STDLAST(fg) = 0;
        } else if (std == FG_STDFIRST(fg)) {
          FG_STDFIRST(fg) = STD_NEXT(std);
        } else if (std == FG_STDLAST(fg)) {
          FG_STDLAST(fg) = STD_PREV(std);
        }
      } else {
        /* turn into labelled continue statement */
        STD_AST(std) = mk_stmt(A_CONTINUE, 0);
      }
    }
    if (laststd)
      break; /* leave loop */
  }
} /* rewritefg */

/*
 * go through loops, innermost first,
 * call findssexprs to build SSlist of all subscript subexpressions
 * used in the loop.
 * call markmodifiedexprs for each loop to mark expressions that
 * are directly modified in this loop
 * call propagatemodified to mark which of these are modified in the loop.
 * call findssexprs to build SSlist of the root subscript subexpressions
 * used in the loop.
 * for each of these, call MakeReplacement to reassociate and distribute,
 * and to find the invariant part of the expression.  Insert an assignment
 * for the invariant part outside of the loop, or move the assignment if the
 * whole expression is invariant.
 */
static void
reassociate(void)
{
  int loop, oldstdavl;
  oldstdavl = astb.std.stg_avail;
  for (loop = 1; loop <= opt.nloops; ++loop) {
    int l, ast, any;
    int fgpre;
    l = LP_LOOP(loop);
    Trace(("REASSOCIATE loop %d: Find all subscript expressions", l));
    findssexprs(l, oldstdavl, SS_ALL, 0, LP_PARLOOP(l));
    Trace(("REASSOCIATE loop %d: Mark modified expressions", l));
    markmodifiedexprs(l, 0);
    Trace(("REASSOCIATE loop %d: propagate modified expressions", l));
    propagatemodified(l);
    Trace(("REASSOCIATE loop %d: find root subscript expressions", l));
    findssexprs(l, oldstdavl, SS_ROOT, 0, LP_PARLOOP(l));
    /* go through each root expression used in this loop.
     * reassociate and distribute for maximum code floating */
    any = 0;
    ast_visit(1, 1);
    for (ast = SSlisthead; ast > 1; ast = A_OPT1G(ast)) {
      /* ast is used in this loop */
      int rast;
      rast = MakeReplacement(ast, l);
      if (rast != ast) {
        ++gany;
        ++any;
        Trace(("REASSOCIATE loop %d, gany=%d: replace ast %d by ast %d", l,
               gany, ast, rast));
#if DEBUG
        if (DBGBIT(39, 4)) {
          printast(ast);
          fprintf(gbl.dbgfil ? gbl.dbgfil : stderr, " ==> ");
          printast(rast);
          fprintf(gbl.dbgfil ? gbl.dbgfil : stderr, "\n");
        }
#endif
        ast_replace(ast, rast);
      }
    }
    if (!any) {
      ast_unvisit();
    } else {
      int fg;
      for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
        int std;
        for (std = FG_STDFIRST(fg); std; std = STD_NEXT(std)) {
          int ast, rast;
          ast = STD_AST(std);
          rast = ast_rewrite(ast);
          if (rast != ast) {
            Trace(("REASSOCIATE loop %d: std %d replace ast %d by ast %d", l,
                   std, ast, rast));
#if DEBUG
            if (DBGBIT(39, 4)) {
              printast(ast);
              fprintf(gbl.dbgfil ? gbl.dbgfil : stderr, " ==> ");
              printast(rast);
              fprintf(gbl.dbgfil ? gbl.dbgfil : stderr, "\n");
            }
#endif
            STD_AST(std) = rast;
          }
          if (std == FG_STDLAST(fg))
            break; /* leave loop */
        }
      }
      ast_unvisit();
      /* now find subscript expressions after changes */
      Trace((
          "REASSOCIATE loop %d: find subscript expressions after reassociation",
          l));
      findssexprs(l, oldstdavl, SS_ALL, 0, LP_PARLOOP(l));
      Trace(("REASSOCIATE loop %d: mark modified subexpressions after "
             "reassociation",
             l));
      markmodifiedexprs(l, 0);
      Trace(("REASSOCIATE loop %d: propagate modified subexpressions after "
             "reassociation",
             l));
      propagatemodified(l);
    }
    Trace(
        ("REASSOCIATE loop %d: find invariant root subscript expressions", l));
    findssexprs(l, oldstdavl, SS_INV_ROOT, 0, LP_PARLOOP(l));
    /* make preheader flow graph node for loop l, where to insert code */
    fgpre = FG_LPREV(LP_HEAD(l));
    opt.pre_fg = add_fg(fgpre);
    FG_PAR(opt.pre_fg) = FG_PAR(fgpre);
    rdilts(opt.pre_fg);
    ast_visit(1, 1);
    any = addssassignments(0, opt.pre_fg, 0, STD_PAR(FG_STDFIRST(LP_HEAD(l))),
                           STD_TASK(FG_STDFIRST(LP_HEAD(l))));
    wrilts(opt.pre_fg);
    if (any) {
      int fg;
      for (fg = LP_FG(l); fg; fg = FG_NEXT(fg)) {
        if (FG_PAR(fg) == LP_PARLOOP(l))
          rewritefg(fg);
      }
    }
#if DEBUG
    if (DBGBIT(39, 4)) {
      dstda();
    }
#endif
    ast_unvisit();
    add_loop_preheader(l);
  }
} /* reassociate */

static void
basic_block_redundant(void)
{
  int fg;
  Trace(("basic_block_redundant()"));
  for (fg = FG_LNEXT(0); fg > 0; fg = FG_LNEXT(fg)) {
    int any, stdfirst, atype;
    Trace(("basic block node %d", fg));
    OnlyFG = fg;
    stdfirst = FG_STDFIRST(fg);
    if (stdfirst == 0)
      continue;
    Trace(("%d=stdfirst", stdfirst));
    Trace(("%d=ast", STD_AST(stdfirst)));
    atype = A_TYPEG(STD_AST(stdfirst));
    Trace(("%d=atype=%s", atype, astb.atypes[atype]));
    /* for purposes of this optimization, pull 'do' out of the basic block */
    if (atype == A_DO || atype == A_MP_PDO) {
      FG_STDFIRST(fg) = STD_NEXT(stdfirst);
    }
    findssexprs(fg, astb.std.stg_avail, SS_ALL_MULTIPLE, fg, -1);
    markmodifiedexprs(fg, fg);
    propagatemodified(fg);
    findssexprs(fg, astb.std.stg_avail, SS_INV_ROOT_MULTIPLE, fg, -1);
    ast_visit(1, 1);
    any = addssassignments(FG_STDFIRST(fg), fg, 1, STD_PAR(FG_STDFIRST(fg)),
                           STD_TASK(FG_STDFIRST(fg)));
    if (any) {
      rewritefg(fg);
    }
    ast_unvisit();
    FG_STDFIRST(fg) = stdfirst;
  }
  OnlyFG = 0;
} /* basic_block_redundant */

static void
Init(void)
{
  InitAST();
  InitSSlist();
  CallAppears = 0;
  if (DT_Assigned.stg_base == NULL) {
    STG_ALLOC_SIDECAR(stb.dt, DT_Assigned);
  }
  STG_CLEAR_ALL(DT_Assigned);
  OnlyFG = 0;
} /* Init */

/*
 * free up any dynamically allocated space
 */
static void
Done(void)
{
  STG_DELETE_SIDECAR(opt.lpb, lpinfo);
  freearea(PSI_AREA);
  optshrd_end();
  STG_DELETE_SIDECAR(stb.dt, DT_Assigned);
} /* Done */

/** \brief Reorder the loops in the LP_LOOP order.
 *
 * Right now they are top-sorted according to the parent relationship,
 * but otherwise unordered.  Here we use a more constructive sort,
 * so a loops children are contiguous to the loop.
 * fill LP_DFN with DFN of loop, and LP_CHILDDFN with DFN of last child.
 */
static void
add_dfn_loop(int l, int *pn)
{
  int ll;
  LP_CHILDDFN(l) = *(pn) + 1;
  for (ll = LP_CHILD(l); ll; ll = LP_SIBLING(ll)) {
    add_dfn_loop(ll, pn);
  }
  ++(*pn);
  LP_DFN(l) = *pn;
  LP_LOOP(*pn) = l;
} /* add_dfn_loop */

void
reorder_dfn_loops()
{
  int n, l;
  STG_ALLOC_SIDECAR(opt.lpb, lpinfo);
  n = 0;
  LP_DFN(0) = 0;
  for (l = LP_CHILD(0); l; l = LP_SIBLING(l)) {
    add_dfn_loop(l, &n);
  }
#if DEBUG
  if (n != opt.nloops) {
    interr("reorder_dfn_loops: wrong number of loops", n, ERR_Severe);
  }
#endif
} /* reorder_dfn_loops */

void
redundss(void)
{
  Trace(("in redundss, func_count=%d", gbl.func_count));
  DT_Assigned.stg_base = NULL;
  optshrd_init();
  flowgraph();
  findloop(HLOPT_ALL);
  gany = 0;
  /* 'fix' the loop order, fill in DFN numbers
   * be the loop order index of the last contained loop */
  reorder_dfn_loops();
#if DEBUG
  if (DBGBIT(39, 2)) {
    dumpfgraph();
    dumploops();
  }
#endif
  Init();
  reassociate();
  if (XBIT(70, 0x200)) {
    Init();
    basic_block_redundant();
  }
  Done();
  Trace(("leaving redundss, func_count=%d, gany=%d", gbl.func_count, gany));
} /* redundss */
