/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
    \file
    \brief This file contains part 2 of the compiler's semantic actions
            (also known as the semant2 phase).
*/

#include "gbldefs.h"
#include "error.h"
#include "gramsm.h"
#include "gramtk.h"
#include "global.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "ast.h"
#include "dinit.h"
#include "rtlRtns.h"

static void common_rel(SST *top, int opc, int tkn_alias);
static void log_negation(SST *top, int tkn_alias);
static void var_ref_list(SST *top, int rhstop);
static void ssa_list(SST *top, int rhstop);
static void ast_cnst(SST *top);
static void ast_conval(SST *top);
static void string_with_kind(SST *top);
static void rewrite_cmplxpart_rval(SST *e);
static int combine_fdiv(int, int, int);
static int has_private(int);
static void form_cmplx_const(SST *, SST *, SST *);
static int reassoc_add(int, int, int);
static int get_mem_sptr_by_name(char *name, int dtype);
static ITEM *mkitem(SST *stkp);
static STD_RANGE *mk_ido_std_range(int prev, int mid, int end);

/**
   \brief semantic actions - part 2.
   \param rednum   reduction number
   \param top      top of stack after reduction
 */
void
semant2(int rednum, SST *top)
{
  int sptr, sptr1, sptr2, dtype;
  int i;
  int opc;
  SST *e1;
  INT rhstop;
  ITEM *itemp, *itemp1;
  STD_RANGE *std_rangep;
  DOINFO *doinfo;
  int dum;
  INT val[2];
  int ast;
  int ast2;
  ACL *aclp;
  char *np;
  int set_aclp;

  switch (rednum) {

  /* ------------------------------------------------------------------ */
  /*
   *      <expression> ::= <primary>   |
   */
  case EXPRESSION1:
    sem.parsing_operator = false;
    break;
  /*
   *      <expression> ::= <addition>  |
   */
  case EXPRESSION2:
    break;
  /*
   *      <expression> ::= <multiplication> |
   */
  case EXPRESSION3:
    break;
  /*
   *      <expression> ::= <exponentiation> |
   */
  case EXPRESSION4:
    break;
  /*
   *      <expression> ::= <disjunction> |
   */
  case EXPRESSION5:
    break;
  /*
   *      <expression> ::= <conjunction> |
   */
  case EXPRESSION6:
    break;
  /*
   *      <expression> ::= <eqv or neqv> |
   */
  case EXPRESSION7:
    break;
  /*
   *      <expression> ::= <log negation> |
   */
  case EXPRESSION8:
    break;
  /*
   *      <expression> ::= <concatenation> |
   */
  case EXPRESSION9:
    break;
  /*
   *      <expression> ::= <relation>     |
   */
  case EXPRESSION10:
    break;
  /*
   *	<expression> ::= <defined binary> |
   */
  case EXPRESSION11:
    break;
  /*
   *	<expression> ::= <defined unary>
   */
  case EXPRESSION12:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <primary> ::= <var ref>  |
   */
  case PRIMARY1:
    /* intercept rval <cplx>%["re"|"im"] and rewrite as a call to I_REAL/I_IMAG
     */
    rewrite_cmplxpart_rval(RHS(1));
    /* cannot allow reference to naked statement function */
    if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED) {
      sptr = SST_SYMG(RHS(1));
      if (STYPEG(sptr) == ST_STFUNC) {
        error(85, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        break;
      }
    }
    if (XBIT(49, 0x400000))
      check_derived_type_array_section(SST_ASTG(RHS(1)));
    SST_PARENP(LHS, 0);
    ast = SST_ASTG(RHS(1));
    if (ast_is_sym(ast)) {
      /* If this <var ref> is a procedure pointer expression, then we
       * need to propagate the dtype from the procedure pointer's interface
       * if it's a function.
       */
      int mem = memsym_of_ast(ast);
      if (is_procedure_ptr(mem)) {
        int iface = 0;
        proc_arginfo(mem, NULL, NULL, &iface);
        if (FVALG(iface) && (dtype = DTYPEG(iface)) ) {
          SST_DTYPEP(LHS, dtype);
        }
     }
    }

    break;
  /*
   *      <primary> ::= <constant> |
   */
  case PRIMARY2:
    SST_IDP(LHS, S_CONST);
    SST_ACLP(LHS, 0); /* prevent UMR */
    SST_PARENP(LHS, 0);
    break;
  /*
   *      <primary> ::= %LOC ( <expression> )
   */
  case PRIMARY3:
    if (flg.standard)
      error(176, 2, gbl.lineno, "%LOC", CNULL);
    sptr2 = SST_SYMG(RHS(3));
    if (sc_local_passbyvalue(sptr2, GBL_CURRFUNC)) {
      /* this is the compiler generated SC_LOCAL,
         ignore the LOC and just return it */

      error(155, 3, gbl.lineno, "unsupported %LOC of VALUE parameter:",
            SYMNAME(sptr2));
    }

    if (mklvalue(RHS(3), 3) == 0)
      fix_term(RHS(3), stb.i0); /* Bad expression */
    SST_DTYPEP(LHS, DT_PTR);
    SST_IDP(LHS, S_EXPR);
    SST_ASTP(LHS, mk_unop(OP_LOC, SST_ASTG(RHS(3)), DT_PTR));
    SST_SHAPEP(LHS, 0);
    (void)sym_mkfunc_nodesc(mkRteRtnNm(RTE_loc), DT_ADDR);
    SST_PARENP(LHS, 0);
    break;
  /*
   *      <primary> ::= <elp> <expression> ) |
   */
  case PRIMARY4:
    *LHS = *RHS(2);
    SST_PARENP(LHS, 1);
    if (XBIT(49, 0x8)) {
      ast2 = SST_ASTG(LHS);
      ast = mk_paren(ast2, SST_DTYPEG(LHS));
      mk_alias(ast, A_ALIASG(ast2));
      SST_ASTP(LHS, ast);
    }
    break;
  /*
   *	<primary> ::= <ac beg> <ac spec> <ac end> |
   */
  case PRIMARY5:
    *LHS = *RHS(2);
    SST_PARENP(LHS, 0);
    break;
  /*
   *	<primary> ::= <substring>
   */
  case PRIMARY6:
    SST_PARENP(LHS, 0);
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <ac beg> ::= '(/'
   */
  case AC_BEG1:
    sem.array_const_level++;
    std_rangep = (STD_RANGE *)getitem(0, sizeof(STD_RANGE));
    std_rangep->start = STD_LAST;
    std_rangep->next = sem.ac_std_range;
    sem.ac_std_range = std_rangep;
    break;
  /* ------------------------------------------------------------------ */
  /*
   *    <ac end> ::= '/)'
   */
  case AC_END1:
    sem.array_const_level--;
    sem.ac_std_range = sem.ac_std_range->next;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<elp> ::= (
   */
  case ELP1:
    if (sem.array_const_level > 0) {
      STD_RECORD *elp = (STD_RECORD *)getitem(0, sizeof(STD_RECORD));
      elp->stkp = RHS(1);
      elp->std = STD_LAST;
      elp->next = sem.elp_stack;
      sem.elp_stack = elp;
    }
    SST_ASTP(LHS, STD_PREV(0));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ac spec> ::= |
   */
  case AC_SPEC1:
    uf("Empty array constructor is not allowed");
    error(155, 3, gbl.lineno,
          "Empty array constructor, expecting type spec and/or ac-value-list",
          "");
    break;
  /*
   *	<ac spec> ::= <ac list> |
   */
  case AC_SPEC2:
    aclp = GET_ACL(15);
    aclp->id = AC_ACONST;
    aclp->next = NULL;
    aclp->subc = (ACL *)SST_BEGG(RHS(1));
    SST_IDP(LHS, S_ACONST);
    SST_ACLP(LHS, aclp);
    SST_SHAPEP(LHS, 0);
    sem.save_aconst = (SST *)LHS;
    SST_DTYPEP(LHS, chk_constructor(aclp, 0));
    break;
  /*
   *	<ac spec> ::= <type spec> :: <ac list> |
   */
  case AC_SPEC3:
    aclp = GET_ACL(15);
    aclp->id = AC_ACONST;
    aclp->next = NULL;
    aclp->subc = (ACL *)SST_BEGG(RHS(3));
    SST_IDP(LHS, S_ACONST);
    SST_ACLP(LHS, aclp);
    SST_SHAPEP(LHS, 0);
    SST_DTYPEP(LHS, chk_constructor(aclp, DDTG(SST_DTYPEG(RHS(1)))));
    break;

  /*
   *	<ac spec> ::= <type spec> ::
   */
  case AC_SPEC4:
    /* Zero-sized array of RHS(1) type */
    SST_IDP(LHS, S_ACONST);
    SST_SHAPEP(LHS, 0);
    SST_ACLP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ac list> ::= <ac list> , <ac item> |
   */
  case AC_LIST1:
    ((ACL *)SST_ENDG(RHS(1)))->next = (ACL *)SST_BEGG(RHS(3));
    SST_ENDP(RHS(1), SST_ENDG(RHS(3)));
    break;
  /*
   *	<ac list> ::= <ac item>
   */
  case AC_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<ac item> ::= <expression> |
   */
  case AC_ITEM1:
    if (SST_IDG(RHS(1)) == S_SCONST || SST_IDG(RHS(1)) == S_ACONST) {
      /* just use existing aclp */
      aclp = SST_ACLG(RHS(1));
    } else {
      e1 = (SST *)getitem(ACL_SAVE_AREA, sizeof(SST));
      *e1 = *RHS(1);
      aclp = GET_ACL(15);
      aclp->id = AC_EXPR;
      aclp->repeatc = aclp->size = 0;
      aclp->next = NULL;
      aclp->subc = NULL;
      aclp->u1.stkp = e1;
    }
    SST_BEGP(LHS, (ITEM *)aclp);
    SST_ENDP(LHS, (ITEM *)aclp);
    goto ac_item_common;
  /*
   *	<ac item> ::= <elp> <ac list> , <implied do control> )
   */
  case AC_ITEM2:
    doinfo = (DOINFO *)SST_BEGG(RHS(4));
    aclp = GET_ACL(15);
    aclp->id = AC_IDO;
    aclp->next = NULL;
    aclp->subc = (ACL *)SST_BEGG(RHS(2));
    aclp->u1.doinfo = doinfo;
    dtype = DTYPEG(doinfo->index_var);
    if (!DT_ISINT(dtype)) {
      if (DT_ISREAL(dtype)) /* non-integer & non-real already detected */
        error(94, 3, gbl.lineno, SYMNAME(doinfo->index_var),
              "- Implied DO index variable");
    }
    SST_BEGP(LHS, (ITEM *)aclp);
    SST_ENDP(LHS, (ITEM *)aclp);
    aclp->u2.std_range = NULL;
    for (STD_RECORD *iter = sem.elp_stack; iter; iter = iter->next) {
      if (iter->stkp == RHS(1)) {
        /* Match the <elp> and mark range of statements generated by
         * loop body or loop control in an implied-do loop.
         */
        sem.elp_stack = iter->next;
        aclp->u2.std_range =
            mk_ido_std_range(iter->std, sem.ac_std_range->mid, STD_LAST);
        break;
      }
    }
    goto ac_item_common;
  /*
   *	<ac item> ::= <expression> : <expression> <opt stride> |
   */
  case AC_ITEM3:
    if (flg.standard)
      error(170, 2, gbl.lineno,
            "Subscript triplet used as an array constructor item", CNULL);
    /*
     * fake by using an 'implied do' array constructor.
     */
    dtype = DT_INT;
    sptr = get_temp(dtype);
    doinfo = get_doinfo(0);
    doinfo->index_var = sptr;
    (void)chk_scalartyp(RHS(1), dtype, FALSE);
    (void)chk_scalartyp(RHS(3), dtype, FALSE);
    if (SST_IDG(RHS(4)) == S_NULL) {
      ast2 = astb.i1;
      doinfo->step_expr = 0;
    } else {
      (void)chk_scalartyp(RHS(4), dtype, FALSE);
      ast2 = doinfo->step_expr = SST_ASTG(RHS(4));
      if (ast2 == astb.i1)
        doinfo->step_expr = 0;
    }
    doinfo->init_expr = SST_ASTG(RHS(1));
    doinfo->limit_expr = SST_ASTG(RHS(3));
    doinfo->count =
        mk_binop(OP_SUB, doinfo->limit_expr, doinfo->init_expr, dtype);
    doinfo->count = mk_binop(OP_ADD, doinfo->count, ast2, dtype);
    doinfo->count = mk_binop(OP_DIV, doinfo->count, ast2, dtype);

    aclp = GET_ACL(15);
    aclp->id = AC_IDO;
    aclp->next = NULL;
    aclp->u1.doinfo = doinfo;

    e1 = (SST *)getitem(ACL_SAVE_AREA, sizeof(SST));
    mkident(e1);
    SST_SYMP(e1, sptr);
    SST_DTYPEP(e1, dtype);
    SST_SHAPEP(e1, 0);
    SST_ASTP(e1, mk_id(sptr));
    aclp->subc = GET_ACL(15);
    aclp->subc->id = AC_EXPR;
    aclp->subc->repeatc = aclp->subc->size = 0;
    aclp->subc->next = NULL;
    aclp->subc->subc = NULL;
    aclp->subc->u1.stkp = e1;

    SST_BEGP(LHS, (ITEM *)aclp);
    SST_ENDP(LHS, (ITEM *)aclp);
    aclp->u2.std_range =
        mk_ido_std_range(sem.ac_std_range->start, sem.ac_std_range->start,
                         STD_LAST);
    goto ac_item_common;
  /*
   *	<ac item> ::= <elp> <ac list> , <expression> )
   */
  case AC_ITEM4:
    /*
     * This hack is to allow parsing an array constructor item which is
     * a complex constant composed of named constants; e.g., ( ONE, TWO ),
     * where ONE and TWO are PARAMETERS.
     */
    aclp = (ACL *)SST_BEGG(RHS(2));
    if (aclp->id == AC_EXPR)
      form_cmplx_const(LHS, aclp->u1.stkp, RHS(4));
    else {
      error(34, 3, gbl.lineno, "(", CNULL);
      SST_IDP(LHS, S_CONST);
      SST_DTYPEP(LHS, DT_INT);
      SST_CVALP(LHS, 0);
      SST_ASTP(LHS, mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS)));
      SST_SHAPEP(LHS, 0);
    }
    e1 = (SST *)getitem(ACL_SAVE_AREA, sizeof(SST));
    *e1 = *LHS;
    aclp = GET_ACL(15);
    aclp->id = AC_EXPR;
    aclp->repeatc = aclp->size = 0;
    aclp->next = NULL;
    aclp->subc = NULL;
    aclp->u1.stkp = e1;
    SST_BEGP(LHS, (ITEM *)aclp);
    SST_ENDP(LHS, (ITEM *)aclp);
  ac_item_common:
    if (sem.array_const_level > 0) {
      /* update for next <ac item> */
      sem.ac_std_range->start = STD_LAST;
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<implied do control> ::= <var ref> <idc eq> <etmp exp> , <etmp exp>
   *<etmp e3>
   */
  case IMPLIED_DO_CONTROL1:
    sptr = mklvalue(RHS(1), 4);
    dtype = DTYPEG(sptr);
    if (!DT_ISREAL(dtype) && !DT_ISINT(dtype)) {
      error(94, 3, gbl.lineno, SYMNAME(sptr), "- Implied DO index variable");
      dtype = DT_INT;
    }
    doinfo = get_doinfo(0);
    doinfo->index_var = sptr;
    if (flg.smp)
      is_dovar_sptr(sptr);
    (void)chk_scalartyp(RHS(3), dtype, FALSE);
    (void)chk_scalartyp(RHS(5), dtype, FALSE);
    if (SST_ASTG(RHS(6)) == 0)
      /* <e3> was not specified */
      SST_ASTP(RHS(6), astb.i1);
    (void)chk_scalartyp(RHS(6), dtype, FALSE);
    doinfo->init_expr = SST_ASTG(RHS(3));
    doinfo->limit_expr = SST_ASTG(RHS(5));
    doinfo->step_expr = SST_ASTG(RHS(6));
    doinfo->count =
        mk_binop(OP_SUB, doinfo->limit_expr, doinfo->init_expr, dtype);
    doinfo->count = mk_binop(OP_ADD, doinfo->count, doinfo->step_expr, dtype);
    doinfo->count = mk_binop(OP_DIV, doinfo->count, doinfo->step_expr, dtype);
    if (DT_ISREAL(dtype))
      doinfo->count = mk_convert(doinfo->count, DT_INT);
    SST_BEGP(LHS, (ITEM *)doinfo);
    /* DOVARP(sptr, 1); do not set flag here, must be done when do is
     * actually used.
     */
    /* Pass up last STD generated before any of the implied expression */
    SST_ASTP(LHS, SST_ASTG(RHS(2)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<idc eq> ::= =
   */
  case IDC_EQ1:
    if (sem.array_const_level > 0) {
      sem.ac_std_range->mid = STD_LAST;
    }
    SST_ASTP(LHS, STD_PREV(0));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<etmp exp> ::= <etmp> <expression>
   */
  case ETMP_EXP1:
    if (sem.etmp_list == NULL) {
      *LHS = *RHS(2);
      sem.use_etmps = FALSE;
    } else {
      ast = check_etmp(RHS(2));
      mkident(LHS);
      SST_SYMP(LHS, A_SPTRG(ast));
      SST_DTYPEP(LHS, A_DTYPEG(ast));
      SST_ASTP(LHS, 0);
    }
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<etmp e3> ::=  |
   */
  case ETMP_E31:
    SST_IDP(LHS, S_CONST);
    SST_CVALP(LHS, 1);
    SST_DTYPEP(LHS, DT_INT);
    SST_ASTP(LHS, 0);
    break;
  /*
   *	<etmp e3> ::= , <etmp exp>
   */
  case ETMP_E32:
    *LHS = *RHS(2);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<etmp> ::=
   */
  case ETMP1:
  /* fall thru */
  /* ------------------------------------------------------------------ */
  /*
   *	<etmp lp> ::= (
   */
  case ETMP_LP1:
    sem.use_etmps = TRUE;
    sem.etmp_list = NULL;
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<var ref list> ::= <var ref list> , <var ref> |
   */
  case VAR_REF_LIST1:
    var_ref_list(top, 3);
    break;
  /*
   *	<var ref list> ::= <var ref>
   */
  case VAR_REF_LIST2:
    var_ref_list(top, 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<dvar ref> ::=   <ident>  |
   */
  case DVAR_REF1:
    sptr = block_local_sym(refsym_inscope((int)SST_SYMG(RHS(1)), OC_OTHER));
    goto var_ref_id_shared;
  /*
   *	<dvar ref> ::= <dvar ref> ( <ssa list> )  |
   */
  case DVAR_REF2:
    SST_TMPP(LHS, 0); /* init in_constr flag for common code */
    goto var_ref_ssa;
  /*
   *	<dvar ref> ::= <dvar ref> . <id> |
   */
  case DVAR_REF3:
    goto var_ref_mem1;
  /*
   *	<dvar ref> ::= <dvar ref> % <id> |
   */
  case DVAR_REF4:
    goto var_ref_mem2;
  /*
   *	<dvar ref> ::= <dvar ref> %LOC
   */
  case DVAR_REF5:
    goto var_ref_mem3;

  /* ------------------------------------------------------------------ */
  /*
   *      <var ref> ::= <ident>  |
   */
  case VAR_REF1:
    /* If we may be processing a type parameter, then
     * do not call refsym and/or process ST_PARAM (when
     * ST_PARAM has the same name as the type parameter) here. We
     * need to keep the expression intact to correctly resolve the
     * length or kind expression.
     */
    sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);
    if (STYPEG(sptr) && sem.type_mode && queue_type_param(sptr, 0, 0, 3)) {
      sptr = insert_sym(sptr);
      STYPEP(sptr, ST_IDENT);
    }
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_PTR && DTY(DTY(dtype + 1)) == TY_PROC) {
      /* Fixup procedure pointer's implicit type for
       * its interface.
       */
      int pp = DTY(dtype + 1);
      int iface = DTY(pp + 2);
      if (iface) {
        DTY(pp + 1) = DTYPEG(iface);
      }
    }
  var_ref_id_shared:
    SST_ACLP(LHS, 0); /* prevent UMR */
    if (STYPEG(sptr) == ST_UNKNOWN && DTY(DTYPEG(sptr)) == TY_DERIVED) {
      /* implicit declaration of derived object ?? */
    }

    if (STYPEG(sptr) == ST_PARAM) {
      /* resolve constant */
      dtype = DTYPEG(sptr);
      if (RUNTIMEG(sptr)) {
        SST_IDP(LHS, S_EXPR);
        SST_DTYPEP(LHS, dtype);
        ast = CONVAL2G(sptr);
      } else if (DTY(dtype) != TY_ARRAY) {
        if (DTY(dtype) == TY_DERIVED && CONVAL1G(sptr) == 0) {
          /* error condition */
          goto usesymbol;
        }
        if (DTY(dtype) == TY_DERIVED) {
          mkident(LHS);
          ast = mk_id(CONVAL1G(sptr));
          aclp = GET_ACL(15);
          aclp->id = AC_AST;
          aclp->u1.ast = ast;
          aclp->dtype = dtype;
          SST_ACLP(LHS, aclp);
        } else {
          SST_IDP(LHS, S_CONST);
        }
        SST_DTYPEP(LHS, dtype);
        SST_SYMP(LHS, CONVAL1G(sptr)); /* get const */
        SST_ERRSYMP(LHS, sptr);        /* save for error tracing */
        ast = mk_id(sptr);
        if (!XBIT(49, 0x10)) /* preserve PARAMETER? */
          ast = A_ALIASG(ast);
      } else {
        /* eventually we may want to create an ACONST, but for
           now ...
         */
        /* use var$ac stored in CONVAL1G(var) */
        mkident(LHS);
        SST_DTYPEP(LHS, dtype);
        SST_SYMP(LHS, CONVAL1G(sptr));
        SST_DBEGP(LHS, 0);
        ast = mk_id(CONVAL1G(sptr));
        aclp = GET_ACL(15);
        aclp->id = AC_AST;
        aclp->u1.ast = ast;
        aclp->dtype = DTYPEG(sptr);
        SST_ACLP(LHS, aclp);
        init_named_array_constant(sptr, gbl.currsub);
        ast = mk_id(sptr);
      }
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    } else { /* resolve these later */
    usesymbol:
      set_aclp = 0;
      if (STYPEG(sptr) == ST_ENTRY || STYPEG(sptr) == ST_PROC)
        /* avoid using PARAMG with these types of symbols --
         * PARAM overlays INMODULE.
         */
        ;
      else if (PARAMG(sptr)) {
        aclp = SST_ACLG(RHS(1));
        set_aclp = 1;
      }
      mkident(LHS);
      SST_SYMP(LHS, sptr);
      if (set_aclp)
        SST_ACLP(LHS, aclp);
      /* Pick up the data type from the symbol table entry which was
       * either: 1) explicitly set by the user, or 2) has the current
       * default value.
       */
      dtype = DTYPEG(sptr);

      if (dtype == DT_NONE) {
        /* This is only okay if identifier is an intrinsic,
         * generic, or predeclared.  This means the function was
         * used as an identifier without parenthesized arguments.
         */
        if (IS_INTRINSIC(STYPEG(sptr)))
          setimplicit(sptr);
        dtype = DTYPEG(sptr);
      }
      SST_DTYPEP(LHS, dtype);
      SST_SHAPEP(LHS, 0);
      SST_ASTP(LHS, 0);
    }
    SST_MNOFFP(LHS, 0);
    SST_PARENP(LHS, 0);
    if (SCG(sptr) == SC_NONE && !scn.is_hpf) /* not in HPF directive*/
      /* actually we may want to set the storage class for
       * executable HPF directives, like redistribute */
      sem_set_storage_class(sptr);
    if (gbl.currsub && PUREG(gbl.currsub)) {
      if (STYPEG(sptr) == ST_ARRAY) {
        if (ENCLFUNCG(sptr) && (DISTG(sptr) || ALIGNG(sptr))) {
          error(155, 3, gbl.lineno,
                "Distributed global data in PURE subprogram is unsupported -",
                SYMNAME(sptr));
        } else if (!ENCLFUNCG(sptr) && !CMBLKG(sptr) && DISTG(sptr)) {
          error(155, 3, gbl.lineno, "Local data in PURE subprogram may be "
                                    "ALIGNED but not DISTRIBUTED -",
                SYMNAME(sptr));
        }
      }
    }
    check_and_add_auto_dealloc(sptr);
    break;
  /*
   *      <var ref> ::= <var primary ssa> ( )  |
   */
  case VAR_REF2:
    sptr = 0;
    if (SST_IDG(RHS(1)) == S_LVALUE || SST_IDG(RHS(1)) == S_EXPR)
      sptr = SST_LSYMG(RHS(1));
    else if (SST_IDG(RHS(1)) == S_DERIVED || SST_IDG(RHS(1)) == S_IDENT)
      sptr = SST_SYMG(RHS(1));
    else if (SST_IDG(RHS(1)) == S_SCONST) {
      (void)mkarg(RHS(1), &dum);
      sptr = SST_SYMG(RHS(1));
    }

    ast = SST_ASTG(RHS(1));
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      sptr1 = memsym_of_ast(ast);
      sptr = BINDG(sptr1);
      break;
    default:
      sptr = 0;
    }
    if (STYPEG(sptr) == ST_PROC && VTOFFG(sptr) && sem.tbp_arg && 
        !NOPASSG(sptr1)) {
      itemp = pop_tbp_arg();
      goto tbp_func_common;
    } else if ((STYPEG(sptr) == ST_USERGENERIC ||
                STYPEG(sptr) == ST_OPERATOR) &&
               VTOFFG(sptr)) {
      if (sem.tbp_arg) { 
#if DEBUG
        assert(!NOPASSG(sptr1), "NOPASS flag set for generic tbp component", 
               sptr1, 3); 
#endif
        itemp = pop_tbp_arg();
        goto var_ref_common;
      } else {
        int dty = TBPLNKG(sptr);
        itemp = ITEM_END;
        if (generic_tbp_has_pass_and_nopass(dty, sptr)) {
          int sp;
          e1 = (SST *)getitem(0, sizeof(SST));
          sp = sym_of_ast(ast);
          SST_SYMP(e1, sp);
          SST_DTYPEP(e1, DTYPEG(sp));
          mkident(e1);
          mkexpr(e1);
          itemp = mkitem(e1);
        }
        goto var_ref_common;
      }
    }
    itemp = ITEM_END;
    goto var_ref_common;

  /*
   *      <var ref> ::= <var primary ssa> ( <ssa list> )  |
   */
  case VAR_REF3:
  var_ref_ssa:
    sptr = 0;
    itemp = SST_BEGG(RHS(3));
    ast = SST_ASTG(RHS(1));
    if (SST_IDG(RHS(1)) == S_LVALUE || SST_IDG(RHS(1)) == S_EXPR)
      sptr = SST_LSYMG(RHS(1));
    else if (SST_IDG(RHS(1)) == S_DERIVED || SST_IDG(RHS(1)) == S_IDENT)
      sptr = SST_SYMG(RHS(1));
    else if (SST_IDG(RHS(1)) == S_SCONST) {
      (void)mkarg(RHS(1), &dum);
      sptr = SST_SYMG(RHS(1));
    }
    if (sptr && STYPEG(sptr) == ST_TYPEDEF && sem.in_struct_constr &&
        sem.param_struct_constr) {
      ACL *aclp;
      int offset, use_keyword, all_set;
      aclp = (ACL *)SST_BEGG(RHS(3));
      if (aclp) {
        if (!sem.new_param_dt) {
          sem.new_param_dt = get_parameterized_dt(DTYPEG(sptr));
          all_set = 0;
        } else {
          all_set = 1;
        }
        use_keyword = 0;

        if (all_set) {
          /* Need to dup the struct constructor tag and assign
           * the new parameterized dtype to it.
           */
          int mem1, mem2, new_sym;
          int tag, dty;

          dty = DTYPEG(sptr);
          tag = DTY(dty + 3);
          new_sym = get_next_sym(SYMNAME(tag), "d");
          DTYPEP(new_sym, sem.new_param_dt);
          defer_pt_decl(sem.new_param_dt, 1);
          for (mem1 = DTY(dty + 1), mem2 = DTY(sem.new_param_dt + 1);
               mem1 > NOSYM && mem2 > NOSYM;
               mem1 = SYMLKG(mem1), mem2 = SYMLKG(mem2)) {
            dup_struct_init(mem2, mem1);
          }
          sem.in_struct_constr = new_sym;
        }

        for (offset = 1; aclp; aclp = aclp->next, ++offset) {
          e1 = aclp->u1.stkp;
          if (e1 && SST_IDG(e1) == S_KEYWORD) {
            SST *e3 = SST_E3G(e1);
            np = scn.id.name + SST_CVALG(e1);
            use_keyword = 1;
            if (!(offset = get_kind_parm_by_name(np, sem.new_param_dt))) {
              sem.param_struct_constr = 2;
              continue;
            } else if (sem.param_struct_constr == 2) {
              error(155, 3, gbl.lineno,
                    "Type parameter keyword may only appear in a"
                    " derived type specification -",
                    np);
            }
            if (all_set)
              continue;

            if (SST_IDG(e3) == S_CONST) {
              put_kind_type_param(sem.new_param_dt, offset, SST_CVALG(e3),
                                  SST_ASTG(e3), 0);
            } else {
              mkexpr(e1);
              put_kind_type_param(sem.new_param_dt, offset, -1, SST_ASTG(e3),
                                  0);
            }

          } else {
            if (use_keyword) {
              error(155, 4, gbl.lineno,
                    "Non keyword= parameter may not follow a keyword="
                    " parameter",
                    NULL);
            }
            if (all_set)
              continue;

            if (e1 && SST_IDG(e1) == S_CONST) {
              put_kind_type_param(sem.new_param_dt, offset, SST_CVALG(e1),
                                  SST_ASTG(e1), 0);
            } else if (e1) {
              mkexpr(e1);
              put_kind_type_param(sem.new_param_dt, offset, -1, SST_ASTG(e1),
                                  0);
            }
          }
        }

        if (!use_keyword && sem.param_struct_constr && is_parameter_context()) {
          sem.param_struct_constr = 2;
        }
      }
    }

    sptr = sptr1 = 0;
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_LABEL:
    case A_ENTRY:
    case A_SUBSCR:
    case A_SUBSTR:
    case A_MEM:
      sptr1 = memsym_of_ast(ast);
      sptr = BINDG(sptr1);
      break;
    }
    if (sptr && (STYPEG(sptr) == ST_PROC || STYPEG(sptr) == ST_USERGENERIC ||
                 STYPEG(sptr) == ST_OPERATOR) &&
        IS_TBP(sptr)) {
      int argno, arg, mem, doif, selector;
      ITEM *itemp2, *curr, *prev;
      SST *sp;
      itemp2 = !NOPASSG(sptr1) ? (ITEM *)pop_tbp_arg() : 0;
      if (!itemp2 && ast) {
        int sp;
        int dty = DTYPEG(pass_sym_of_ast(ast));
        if (generic_tbp_has_pass_and_nopass(dty, sptr)) {
          int mem2;
          int iface, paramct, arg_cnt;

          mem2 = get_generic_tbp_pass_or_nopass(dty, sptr, 1);
          proc_arginfo(VTABLEG(mem2), &paramct, 0, &iface);
          for (arg_cnt = 0, itemp2 = itemp; itemp2 != ITEM_END;
               itemp2 = itemp2->next) {
            ++arg_cnt;
          }
          if (arg_cnt >= paramct) {
            sptr = get_generic_tbp_pass_or_nopass(dty, sptr, 0);
            sptr = VTABLEG(sptr);
            goto tbp_func_common;
          }
          e1 = (SST *)getitem(0, sizeof(SST));
          sp = sym_of_ast(ast);
          SST_SYMP(e1, sp);
          SST_DTYPEP(e1, DTYPEG(sp));
          mkident(e1);
          mkexpr(e1);
          itemp2 = mkitem(e1);
          sptr = get_generic_tbp_pass_or_nopass(dty, sptr, 0);
          sptr = VTABLEG(sptr);
        } else if (NOPASSG(sptr1)) {
          goto var_ref_common; /* assume NOPASS tbp */
        } else {
          e1 = (SST *)getitem(0, sizeof(SST));
          sp = pass_sym_of_ast(ast);
          SST_SYMP(e1, sp);
          SST_DTYPEP(e1, DTYPEG(sp));
          mkident(e1);
          mkexpr(e1);
          itemp2 = mkitem(e1);
          if (pass_sym_of_ast(ast) != sym_of_ast(ast)) {
            int a = mk_member(A_PARENTG(ast), mk_id(memsym_of_ast(ast)), dty);
            SST_ASTP(itemp2->t.stkp, a);
            A_DTYPEP(a, DTYPEG(memsym_of_ast(ast)));
            ast = a;
          }
        }
      }
      if (itemp2) {
        if ((STYPEG(sptr) == ST_USERGENERIC || STYPEG(sptr) == ST_OPERATOR)) {
          ITEM *itemp2;
          int arg_cnt;
          int mem, mem2;
          for (arg_cnt = 0, itemp2 = itemp; itemp2 != ITEM_END;
               itemp2 = itemp2->next) {
            ++arg_cnt;
          }
          mem = get_generic_member2(TBPLNKG(sptr), sptr, arg_cnt, NULL);
          mem2 = get_specific_member(TBPLNKG(sptr), VTABLEG(mem));
          argno = get_tbp_argno(BINDG(mem2), TBPLNKG(sptr));
          if (!argno && NOPASSG(mem2)) {
            goto var_ref_common; /* assume NOPASS tbp */
          }
        } else {
          argno = get_tbp_argno(sptr, DTYPEG(pass_sym_of_ast(ast)));
          if (!argno && NOPASSG(sptr1)) {
            goto var_ref_common; /* assume NOPASS tbp */
          }
        }
        if (!argno)
          break; /* error -- probably no interface specified */

        if (itemp == ITEM_END) {
          itemp = itemp2;
          itemp->next = ITEM_END;
        } else {
          for (arg = 1, curr = prev = itemp; arg <= argno;) {
            if (arg == argno) {
              itemp2->next = curr;
              if (argno == 1) {
                itemp = itemp2;
              } else {
                prev->next = itemp2;
              }
              break;
            }
            ++arg;
            prev = curr;
            if (curr == ITEM_END) {
              interr("semant2: bad item list for <cvar ref> ", 0, 3);
              break;
            }
            curr = curr->next;
          }
        }
      }
    tbp_func_common:
      doif = sem.doif_depth;
      selector = pass_sym_of_ast(ast);
      dtype = DTYPEG(selector);
      if (DTY(dtype) == TY_ARRAY)
        dtype = DTY(dtype + 1);
      argno = get_tbp_argno(sptr, dtype);
      for (sp = 0, arg = 1, curr = itemp; curr != ITEM_END; curr = curr->next) {
        if (arg == argno) {
          sp = itemp->t.stkp;
          break;
        }
        ++arg;
      }
      sptr1 = 0;
      if (SST_IDG(sp) == S_LVALUE || SST_IDG(sp) == S_EXPR)
        sptr1 = SST_LSYMG(sp);
      else if (SST_IDG(sp) == S_DERIVED || SST_IDG(sp) == S_IDENT)
        sptr1 = SST_SYMG(sp);
      else if (SST_IDG(sp) == S_SCONST) {
        (void)mkarg(sp, &dum);
        sptr1 = SST_SYMG(sp);
      }
      dtype = DTYPEG(sptr1);
      if (DTY(dtype) == TY_ARRAY)
        dtype = DTY(dtype + 1);
      sptr = get_implementation(dtype, sptr, 0, &mem);
      if (0 && CLASSG(sptr1)) {
        CLASSP(sptr, 1);
      }
      SST_SYMP(RHS(1), sptr);
      SST_LSYMP(RHS(1), sptr);
      SST_DTYPEP(RHS(1), DTYPEG(sptr));
    }
  var_ref_common:

    if (SST_IDG(RHS(1)) == S_CONST)
      goto variable_reference;

    if (sem.in_struct_constr) {
      if (sem.param_struct_constr == 1) {
        break;
      }
      sptr = SST_SYMG(RHS(1));
      i = ENCLFUNCG(sptr);
      if (i && STYPEG(i) == ST_MODULE) {
        /* type was defined in a module.  If we are
           not in a module procedure of that module,
           and if the type is private, or all components
           are private, we want to give an error */

        if (!sem.mod_cnt && has_private(DTYPEG(sptr)))
          error(155, 3, gbl.lineno,
                "Using structure constructor for type with private components:",
                SYMNAME(sptr));
      }
      if (SST_IDG(RHS(1)) == S_SCONST)
        /* previous subscript was encountered: error */
        error(155, 3, gbl.lineno, "Bad structure constructor()() - type",
              SYMNAME(SST_SYMG(RHS(1))));
      else if (itemp == ITEM_END &&
               (!no_data_components(DTYPEG(sem.in_struct_constr)) &&
                !all_default_init(DTYPEG(sem.in_struct_constr))))
        error(155, 4, gbl.lineno, "Empty structure constructor() - type",
              SYMNAME(SST_SYMG(RHS(1))));
      else if (sem.in_stfunc) {
        error(155, 3, gbl.lineno,
              "Structure constructor not allowed in statement function",
              SYMNAME(SST_SYMG(RHS(1))));
        sem.stfunc_error = TRUE;
      }
      if (is_empty_typedef(DTYPEG(sem.in_struct_constr)) && itemp != ITEM_END) {
        /* Handle empty typedef */
        error(155, 3, gbl.lineno, "Structure constructor specified"
                                  " for empty derived type",
              SYMNAME(SST_SYMG(RHS(1))));
      }
      dtype = DTYPEG(sem.in_struct_constr);

      if (itemp == ITEM_END &&
         (aclp = all_default_init(DTYPEG(sem.in_struct_constr)))) {
        /* Initialize the empty structure constructor with
         * the first default initializer...
         */
        ACL *a;
        a = GET_ACL(15);
        a->id = AC_SCONST;
        a->next = NULL;
        a->subc = aclp;
        a->dtype = dtype = DTYPEG(sem.in_struct_constr);
        SST_IDP(LHS, S_SCONST);
        SST_DTYPEP(LHS, dtype);
        SST_ACLP(LHS, a);
        chk_struct_constructor(a);
        SST_SYMP(LHS, sem.in_struct_constr);
        sem.in_struct_constr = SST_TMPG(LHS); /*restore old value */
        break;
      }

      /* create head AC_SCONST for element list */

      aclp = GET_ACL(15);
      aclp->id = AC_SCONST;
      aclp->next = NULL;
      aclp->dtype = dtype = DTYPEG(sem.in_struct_constr);
      if (no_data_components(dtype)) {
        aclp->subc = NULL;
      } else {
        aclp->subc = (ACL *)SST_BEGG(RHS(3));
        chk_struct_constructor(aclp);
      }
      SST_IDP(LHS, S_SCONST);
      SST_DTYPEP(LHS, dtype);
      SST_ACLP(LHS, aclp);
      SST_SYMP(LHS, sem.in_struct_constr);
      sem.in_struct_constr = SST_TMPG(LHS); /*restore old value */
      break;
    }
    sem.in_struct_constr = SST_TMPG(LHS); /* restore old value */
                                          /*
                                           * Must be careful here.  If <var primary> is S_IDENT, then
                                           * this is a potential statement function definition.  Use
                                           * the S_STFUNC stack type to delay processing.
                                           */
    if (sem.psfunc &&                     /* left side of assgnmt */
        sem.pgphase < PHASE_EXEC &&       /* no exec. stmts yet   */
        sem.tkntyp == TK_EQUALS &&        /* '=' causes reduction */
        SST_IDG(RHS(1)) == S_IDENT &&     /* simple identifier    */
        DTY(SST_DTYPEG(RHS(1))) != TY_ARRAY &&
        STYPEG(SST_SYMG(RHS(1))) != ST_ENTRY) {
      LOGICAL stfunc_error = FALSE;
      /* This could be a statement function definition.
       * If the ssa list is of the form (<ident> = <expression> OR
       * e1:e2:e3 OR a constant) then this cannot be considered a
       * statement function
       */
      for (itemp1 = itemp; (itemp1 && itemp1 != ITEM_END);
           itemp1 = itemp1->next) {
        e1 = itemp1->t.stkp;
        if (SST_IDG(e1) == S_TRIPLE || SST_IDG(e1) == S_CONST ||
            SST_IDG(e1) == S_KEYWORD || SST_IDG(e1) == S_LABEL) {
          goto variable_reference;
        }
      }
      /*  Have a statement function; perform error checking on the
       *  arguments.
       */
      if (sem.block_scope)
        error(1218, ERR_Severe, gbl.lineno, "A statement function", CNULL);
      for (itemp1 = itemp; itemp1 != ITEM_END; itemp1 = itemp1->next) {
        e1 = itemp1->t.stkp;
        /*  check that only identifier names appeared between "()" */
        if (SST_IDG(e1) != S_IDENT) {
          errsev(86);
          stfunc_error = TRUE;
          break;
        }
        sptr = SST_SYMG(e1);
        switch (STYPEG(sptr)) {
        case ST_UNKNOWN:
        case ST_IDENT:
        case ST_VAR:
          break;
        case ST_INTRIN:
        case ST_GENERIC:
        case ST_PD:
          if (!EXPSTG(sptr))
            break;
          FLANG_FALLTHROUGH;
        default:
          error(84, 3, gbl.lineno, SYMNAME(sptr),
                "as a dummy argument to a statement function");
          stfunc_error = TRUE;
        }
        if (stfunc_error) {
          break;
        }
        sptr = declsym(sptr, ST_VAR, FALSE);
        SST_SYMP(e1, sptr);
        SST_ASTP(e1, mk_id(sptr));
        SST_SHAPEP(e1, A_SHAPEG(SST_ASTG(e1)));
        dtype = DTYPEG(sptr);
        if (dtype == DT_ASSCHAR || dtype == DT_ASSNCHAR ||
            dtype == DT_DEFERCHAR || dtype == DT_DEFERNCHAR) {
          error(89, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          stfunc_error = TRUE;
          break;
        }
      }
      sem.stfunc_error = stfunc_error;
      sem.in_stfunc = TRUE;
      sptr = SST_SYMG(RHS(1));
      SST_IDP(LHS, S_STFUNC);
      SST_SYMP(LHS, sptr);
      SST_ENDP(LHS, itemp);
      if (DTYPEG(sptr) == DT_ASSCHAR || DTYPEG(sptr) == DT_ASSNCHAR ||
          DTYPEG(sptr) == DT_DEFERCHAR || DTYPEG(sptr) == DT_DEFERNCHAR) {
        error(89, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        sem.stfunc_error = TRUE;
      }
    } else if (DTY(SST_DTYPEG(RHS(1))) == TY_ARRAY) { /* ptr reshape */
      /* If this is a strided array reference, then save which
       * parts of the section the user specified in a bit mask.
       * We do this since ref_array() called by mkvarref() will replace
       * any omitted parts with an expression (e.g., p(1:) gets replaced
       * with p(1:ubound(p))). While this works for most pointer/array
       * sections, it doesn't work if the section is the destination of
       * an assignment. In the case of pointer reshaping, we could have this
       * type of expression on the left hand side. By saving this info
       * we can properly perform semantic checking as well as compute
       * the correct ubound through the run-time library.
       */
      int triple_flag, currDim;
      int empty;
      empty = 1;
      for (triple_flag = currDim = 0, itemp1 = itemp;
           (itemp1 && itemp1 != ITEM_END); itemp1 = itemp1->next) {
        e1 = itemp1->t.stkp;
        if (SST_IDG(e1) == S_TRIPLE) {
          int mask;
          mask = 0;
          if (SST_IDG(SST_E1G(e1)) == S_NULL) {
            mask |= lboundMask;
          }
          if (SST_IDG(SST_E2G(e1)) == S_NULL) {
            mask |= uboundMask;
          }
          if (SST_IDG(SST_E3G(e1)) == S_NULL) {
            mask |= strideMask;
          }
          if (empty && mask != (lboundMask | uboundMask | strideMask))
            empty = 0;
          mask <<= 3 * currDim;
          triple_flag |= mask;
          currDim++;
        } else {
          empty = 0;
        }
      }
      SST_DIMFLAGP(LHS, triple_flag);
      goto variable_reference;
    } else {
    variable_reference:
      (void)mkvarref(RHS(1), itemp);
    }
    if (SST_IDG(RHS(1)) == S_EXPR) {
      /* Needed by <tpv> ::= <expression> in semant.c */
      ast = SST_ASTG(RHS(1));
      switch (A_TYPEG(ast)) {
      case A_ID:
      case A_LABEL:
      case A_ENTRY:
      case A_SUBSCR:
      case A_SUBSTR:
      case A_MEM:
        sptr = memsym_of_ast(ast);
        break;
      default:
        sptr = 0;
      }
      SST_LSYMP(LHS, sptr);
    }
    SST_PARENP(LHS, 0);
    break;
  /*
   *      <var ref> ::= <var primary> . <id>
   */
  case VAR_REF4:
  var_ref_mem1:
    (void)mkexpr(RHS(1));
    dtype = SST_DTYPEG(RHS(1));
    if (DTY(DDTG(dtype)) == TY_DERIVED) {
      if (!sem.generic_tbp) {
        /* check for generic type bound procedure  for
         * defined binary operator
         */
        int mem;
        mem = 0;
        get_implementation(DDTG(dtype), SST_SYMG(RHS(3)), 0, &mem);
        if (mem) {
          sem.generic_tbp = BINDG(mem);
          break;
        }
      } else {
        /* resolve generic type bound procedure for defined binary
         * operator */
        sptr1 = SST_SYMG(RHS(3));
        dtype = DTYPEG(sptr1);
        if (dtype)
          SST_DTYPEP(RHS(3), dtype);
        SST_ASTP(RHS(3), mk_id(sptr1));
        defined_operator(sem.generic_tbp, LHS, RHS(1), RHS(3));
        sem.generic_tbp = 0;
        SST_PARENP(LHS, 0);
        break;
      }
    } else if (!sem.generic_tbp) {
      char *name = SYMNAME(SST_SYMG(RHS(3)));
      int sym = findByNameStypeScope(name, ST_OPERATOR, 0);
      if (sym > NOSYM && CLASSG(sym) && IS_TBP(sym)) {
        sem.generic_tbp = sym;
        break;
      } else if (sym > NOSYM || sem.parsing_operator) {
        /* If sym > NOSYM then we are parsing the beginning of a user defined
         * operator. If sem.parsing_operator is true and sym <= NOSYM, then
         * we are parsing the end of the operator.
         */
        sem.parsing_operator = (sym > NOSYM);
        break;
      }
    } else {
      /* resolve generic type bound procedure for defined binary
       * operator */
      sptr1 = SST_SYMG(RHS(3));
      dtype = DTYPEG(sptr1);
      if (dtype)
        SST_DTYPEP(RHS(3), dtype);
      SST_ASTP(RHS(3), mk_id(sptr1));
      defined_operator(sem.generic_tbp, LHS, RHS(1), RHS(3));
      sem.generic_tbp = 0;
      SST_PARENP(LHS, 0);
      break;
    }

    if (flg.standard)
      error(179, 2, gbl.lineno, SYMNAME(DTY(dtype + 1)), CNULL);
    if (DTY(dtype) != TY_STRUCT) {
      error(141, 3, gbl.lineno, "RECORD", ".");
      break;
    }
    sptr1 = SST_SYMG(RHS(3));
    i = NMPTRG(sptr1);
    ast = SST_ASTG(RHS(1));
    ast = mkmember(dtype, ast, i);
    if (ast) {
      sptr = A_SPTRG(A_MEMG(ast));
      dtype = DTYPEG(sptr);
      SST_IDP(LHS, S_LVALUE);
      SST_LSYMP(LHS, sptr);
      SST_SHAPEP(LHS, 0);
      SST_DTYPEP(LHS, dtype);
    } else {
      /* <id> is not a member of this record */
      error(142, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
      SST_IDP(LHS, S_LVALUE);
      SST_LSYMP(LHS, sptr1);
      SST_DTYPEP(LHS, DT_INT);
    }
    SST_ASTP(LHS, ast);
    SST_SHAPEP(LHS, A_SHAPEG(ast));
    SST_PARENP(LHS, 0);
    break;

  /*
   *	<var ref> ::= <var primary> % <id> |
   */
  case VAR_REF5:
  var_ref_mem2:
    (void)mkexpr(RHS(1));
    rhstop = 3;
    goto var_ref_component_shared;

  /*
   *	<var ref> ::= <var primary> %LOC
   */
  case VAR_REF6:
  var_ref_mem3:
    (void)mkexpr(RHS(1));
    rhstop = 2;
    SST_SYMP(RHS(2), getsymbol("loc"));
  var_ref_component_shared:
    if (SST_IDG(RHS(1)) == S_IDENT || SST_IDG(RHS(1)) == S_DERIVED)
      sptr = SST_SYMG(RHS(1));
    else {
      if (SST_IDG(RHS(1)) == S_EXPR && A_TYPEG(SST_ASTG(RHS(1))) == A_FUNC) {
        error(155, 3, gbl.lineno,
              "Illegal context for the component reference to",
              SYMNAME(SST_SYMG(RHS(rhstop))));
      }
      sptr = SST_LSYMG(RHS(1));
    }
    dtype = DTYPEG(sptr);
    if (DTY(dtype) == TY_ARRAY)
      dtype = DTY(dtype + 1);
    if (DTY(dtype) == TY_DERIVED) {
      int orig_mem_sptr = SST_SYMG(RHS(rhstop));
      int mem_sptr = orig_mem_sptr;
      int tbp_sptr, mem;
      if (mem_sptr > NOSYM)
        mem_sptr = sym_skip_construct(mem_sptr);
      if (mem_sptr > NOSYM && STYPEG(mem_sptr) != ST_PROC &&
          STYPEG(mem_sptr) != ST_ENTRY && STYPEG(mem_sptr) != ST_USERGENERIC &&
          get_implementation(dtype, mem_sptr, 0, &mem) && mem > NOSYM &&
          BINDG(mem) > NOSYM)
        mem_sptr = BINDG(mem);
      if (mem_sptr <= NOSYM ||
          (STYPEG(mem_sptr) != ST_PROC && STYPEG(mem_sptr) != ST_ENTRY &&
           STYPEG(mem_sptr) != ST_USERGENERIC))
        goto normal_var_ref_component;
      ast = SST_ASTG(RHS(1));
      if (A_ORIG_EXPRG(ast) != 0) {
        /* This is a type bound procedure, so restore original expression. */
        ast = A_ORIG_EXPRG(ast);
        SST_ASTP(RHS(1), ast);
      }
      switch (A_TYPEG(SST_ASTG(RHS(1)))) {
      case A_ID:
      case A_LABEL:
      case A_ENTRY:
      case A_SUBSCR:
      case A_SUBSTR:
      case A_MEM:
        break;
      default:
        goto normal_var_ref_component;
      }
      sptr = memsym_of_ast(SST_ASTG(RHS(1)));
      dtype = DTYPEG(sptr);
      if (DTY(dtype) == TY_ARRAY)
        dtype = DTY(dtype + 1);
      if (get_implementation(dtype, mem_sptr, 0, &mem) && mem > NOSYM &&
          (tbp_sptr = BINDG(mem)) > NOSYM &&
          (STYPEG(tbp_sptr) == ST_PROC || STYPEG(tbp_sptr) == ST_USERGENERIC) &&
          IS_TBP(tbp_sptr)) {
        if (!NOPASSG(mem)) {
          SST *e1 = (SST *)getitem(0, sizeof *e1);
          *e1 = *RHS(1);
          push_tbp_arg(mkitem(e1));
        }
        ast = SST_ASTG(RHS(1));
        ast = mkmember(dtype, ast, NMPTRG(mem));
        /* get_implementation(dtype, VTABLEG(mem), 0, &mem); */
        if (ast) {
          SST_IDP(LHS, S_LVALUE);
          SST_LSYMP(LHS, tbp_sptr);
          SST_SYMP(LHS, tbp_sptr);
          SST_SHAPEP(LHS, 0);
          SST_DTYPEP(LHS, A_DTYPEG(ast));
          SST_ASTP(LHS, ast);
          SST_PARENP(LHS, 0);
        } else {
          error(142, 3, gbl.lineno, SYMNAME(orig_mem_sptr), CNULL);
        }
        break;
      }
    } else if (DT_ISCMPLX(dtype) && (sptr1 = SST_SYMG(RHS(rhstop)))) {
      if (strcmp(SYMNAME(sptr1), "re") == 0 ||
          strcmp(SYMNAME(sptr1), "im") == 0) {
        /* build a phoney member ast that will be rewritten later */
        dtype = DTY(dtype) == TY_CMPLX  ? DT_REAL4
#ifdef TARGET_SUPPORTS_QUADFP
              : DTY(dtype) == TY_QCMPLX ? DT_QUAD
#endif
                                        : DT_REAL8;
        STYPEP(sptr1, ST_MEMBER);
        DTYPEP(sptr1, dtype); /* don't count on this, it will change */
        SST_ASTP(LHS, mk_member(SST_ASTG(RHS(1)), mk_id(sptr1), dtype));
        SST_ACLP(LHS, 0);
        SST_PARENP(LHS, 0);
        break;
      }
    }
  normal_var_ref_component:
    dtype = SST_DTYPEG(RHS(1));
    dtype = DDTG(dtype);
    ast = SST_ASTG(RHS(1));
    if (DTY(dtype) != TY_DERIVED) {
      error(141, 3, gbl.lineno, "Derived Type object", "%");
      break;
    }
    i = NMPTRG(SST_SYMG(RHS(rhstop)));
    ast = mkmember(dtype, ast, i);
    if (ast) {
      sptr1 = A_SPTRG(A_MEMG(ast));
      if (PRIVATEG(sptr1) && test_private_dtype(ENCLDTYPEG(sptr1))) {
        error(155, 3, gbl.lineno, "Attempt to use private component:",
              SYMNAME(sptr1));
      }

      /*dtype = DTYPEG(sptr1);*/
      dtype = A_DTYPEG(ast);
      SST_IDP(LHS, S_LVALUE);
      SST_LSYMP(LHS, sptr1);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
      SST_DTYPEP(LHS, dtype);
    } else {
      /* <id> is not a member of this record */
      sptr1 = SST_SYMG(RHS(rhstop));
      error(142, 3, gbl.lineno, SYMNAME(sptr1), CNULL);
      SST_IDP(LHS, S_LVALUE);
      SST_LSYMP(LHS, sptr1);
      SST_DTYPEP(LHS, DT_INT);
    }
    SST_ACLP(LHS, 0);
    SST_ASTP(LHS, ast);
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <var primary ssa> ::= <var primary>
   */
  case VAR_PRIMARY_SSA1:
    if (SST_IDG(RHS(1)) == S_CONST) {
      sem.in_struct_constr = 0;
    } else {
      sptr = SST_SYMG(RHS(1));
      dtype = DTYPEG(sptr);

      SST_TMPP(LHS, sem.in_struct_constr); /* save old value */
      /* set a flag for ssa list processing */
      if (STYPEG(sptr) == ST_TYPEDEF && DTY(dtype) == TY_DERIVED) {
        sem.in_struct_constr = sptr;
        if (has_type_parameter(dtype))
          sem.param_struct_constr += 1;
      } else
        sem.in_struct_constr = 0;
    }
    break;
  /*
   *      <var primary> ::= <var ref>
   */
  case VAR_PRIMARY1:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <ssa list> ::= <ssa list> , <ssa> |
   */
  case SSA_LIST1:
    ssa_list(top, 3);
    break;
  /*
   *      <ssa list> ::= <ssa>
   */
  case SSA_LIST2:
    ssa_list(top, 1);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <ssa> ::= <expression>  |
   */
  case SSA1:
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    *e1 = *RHS(1);
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;
  /*
   *      <ssa> ::= <id name> = <expression>  |
   */
  case SSA2:
    if (SST_IDG(RHS(3)) == S_ACONST && sem.in_struct_constr &&
        sem.save_aconst) {
      i = get_mem_sptr_by_name(scn.id.name + SST_CVALG(RHS(1)),
                               DTYPEG(sem.in_struct_constr));
      if (i) {
        /* Make sure element type of array constructor matches
         * element type of struct member.
         */
        int d = DTYPEG(i);
        e1 = sem.save_aconst;
        chk_constructor(SST_ACLG(e1), DTY(d + 1));
        *RHS(3) = *e1;
        sem.save_aconst = 0;
      }
    }
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    *e1 = *RHS(1);
    SST_IDP(e1, S_KEYWORD);
    SST_E1P(LHS, e1);
    SST_E3P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E3G(e1)) = *RHS(3);
    SST_E2P(LHS, 0);
    break;
  /*
   *      <ssa> ::= <opt sub> : <opt sub> <opt stride> |
   */
  case SSA3:
    /* Build a triplet ssa list entry */
    e1 = (SST *)getitem(sem.ssa_area, sizeof(SST));
    SST_IDP(e1, S_TRIPLE);
    SST_E1P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E1G(e1)) = *RHS(1);
    SST_E2P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E2G(e1)) = *RHS(3);
    SST_E3P(e1, (SST *)getitem(sem.ssa_area, sizeof(SST)));
    *(SST_E3G(e1)) = *RHS(4);
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;
  /*
   *	<ssa> ::= <arg builtin> |
   */
  case SSA4:
    break;
  /*
   *	<ssa> ::= * <reflabel> |
   */
  case SSA5:
    goto ssa_lab_shared;
  /*
   *	<ssa> ::= & <reflabel>
   */
  case SSA6:
    if (flg.standard)
      errwarn(181);
  ssa_lab_shared:
    e1 = (SST *)getitem(0, sizeof(SST));
    *e1 = *RHS(2);
    SST_E1P(LHS, e1);
    SST_IDP(e1, S_LABEL);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel data ss> ::= <accel data name> ( <accel sub list> )
   */
  case ACCEL_DATA_SS1:
  accel_data_ss1:
    itemp = SST_BEGG(RHS(3));
    (void)mkvarref(RHS(1), itemp);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel data name> ::= <ident> |
   */
  case ACCEL_DATA_NAME1:
  accel_data_name1:
    /*sptr = refsym((int)SST_SYMG(RHS(1)), OC_OTHER);*/
    sptr = SST_SYMG(RHS(1));
    sptr = find_outer_sym(sptr);
    SST_ACLP(LHS, 0); /* prevent UMR */
    mkident(LHS);
    SST_SYMP(LHS, sptr);
    SST_SHAPEP(LHS, 0);
    SST_ASTP(LHS, 0);
    SST_MNOFFP(LHS, 0);
    SST_PARENP(LHS, 0);
    break;
  /*
   *	<accel data name> ::= <accel data name> % <id> |
   */
  case ACCEL_DATA_NAME2:
  accel_data_name2:
    goto var_ref_mem2;
  /*
   *	<accel data name> ::= <accel data ss> % <id>
   */
  case ACCEL_DATA_NAME3:
  accel_data_name3:
    goto var_ref_mem2;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl data ss> ::= <accel decl data name> ( <accel decl sub list>
   *)
   */
  case ACCEL_DECL_DATA_SS1:
    goto accel_data_ss1;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdecl data ss> ::= <accel mdecl data name> ( <accel decl sub
   *list> )
   */
  case ACCEL_MDECL_DATA_SS1:
    goto accel_data_ss1;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel decl data name> ::= <ident> |
   */
  case ACCEL_DECL_DATA_NAME1:
    goto accel_data_name1;
  /*
   *	<accel decl data name> ::= <accel decl data name> % <id> |
   */
  case ACCEL_DECL_DATA_NAME2:
    goto accel_data_name2;
  /*
   *	<accel decl data name> ::= <accel decl data ss> % <id>
   */
  case ACCEL_DECL_DATA_NAME3:
    goto accel_data_name3;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdecl data name> ::= <ident> |
   */
  case ACCEL_MDECL_DATA_NAME1:
    goto accel_data_name1;
  /*
   *	<accel mdecl data name> ::= <accel mdecl data name> % <id> |
   */
  case ACCEL_MDECL_DATA_NAME2:
    goto accel_data_name2;
  /*
   *	<accel mdecl data name> ::= <accel mdecl data ss> % <id>
   */
  case ACCEL_MDECL_DATA_NAME3:
    goto accel_data_name3;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdata ss> ::= <accel mdata name> ( <accel sub list> )
   */
  case ACCEL_MDATA_SS1:
    goto accel_data_ss1;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel mdata name> ::= <ident> |
   */
  case ACCEL_MDATA_NAME1:
    goto accel_data_name1;
  /*
   *	<accel mdata name> ::= <accel mdata name> % <id> |
   */
  case ACCEL_MDATA_NAME2:
    goto accel_data_name2;
  /*
   *	<accel mdata name> ::= <accel mdata ss> % <id>
   */
  case ACCEL_MDATA_NAME3:
    goto accel_data_name3;

  /* ------------------------------------------------------------------ */
  /*
   *	<accel sdata name> ::= <ident>
   */
  case ACCEL_SDATA_NAME1:
    goto accel_data_name1;

  /* ------------------------------------------------------------------ */
  /*
   *      <arg builtin> ::= % <id name> ( <expression> )
   */
  case ARG_BUILTIN1:
    e1 = (SST *)getitem(0, sizeof(SST));
    np = scn.id.name + SST_CVALG(RHS(2));
    if (sem_strcmp(np, "val") == 0) {
      if (flg.standard)
        error(176, 2, gbl.lineno, "%VAL", CNULL);
      *e1 = *RHS(4);
      if (SST_ISNONDECC(e1))
        cngtyp(e1, DT_INT);
      mkexpr(e1);
      SST_IDP(e1, S_VAL);
      dtype = SST_DTYPEG(RHS(4));
      if (!DT_ISBASIC(dtype) && DTY(dtype) != TY_STRUCT &&
          DTY(dtype) != TY_DERIVED) {
        cngtyp(e1, DT_INT);
        errsev(52);
      }
      SST_ASTP(e1, mk_unop(OP_VAL, SST_ASTG(e1), dtype));
    } else if (sem_strcmp(np, "ref") == 0) {
      if (flg.standard)
        error(176, 2, gbl.lineno, "%REF", CNULL);
      switch (SST_IDG(RHS(4))) {
      case S_REF:
      case S_VAL:
        errsev(53);
        SST_IDP(e1, S_CONST);
        SST_DTYPEP(e1, DT_INT);
        SST_CVALP(e1, 0);
        break;
      default:
        mkarg(RHS(4), &dum);
        *e1 = *RHS(4);
        SST_IDP(e1, S_REF);
        break;
      }
      SST_ASTP(e1, mk_unop(OP_REF, SST_ASTG(RHS(4)), DT_INT));
    } else {
      error(34, 3, gbl.lineno, np, CNULL);
      SST_IDP(LHS, S_CONST);
      SST_CVALP(LHS, 0);
      SST_DTYPEP(LHS, DT_INT);
      SST_ASTP(LHS, astb.i0);
      *e1 = *LHS;
    }
    SST_E1P(LHS, e1);
    SST_E2P(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <opt sub> ::= |
   */
  case OPT_SUB1:
    SST_IDP(LHS, S_NULL);
    break;
  /*
   *      <opt sub> ::= <expression>
   */
  case OPT_SUB2:
    break;
  /*
   *      <opt stride> ::= |
   */
  case OPT_STRIDE1:
    SST_IDP(LHS, S_NULL);
    SST_ASTP(LHS, 0);
    break;
  /*
   *      <opt stride> ::= : <expression>
   */
  case OPT_STRIDE2:
    *LHS = *RHS(2);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <constant> ::= <integer> |
   */
  case CONSTANT1:
    if (DTY(stb.user.dt_int) == TY_INT8) {
      if ((INT)SST_CVALG(RHS(1)) < 0)
        val[0] = -1;
      else
        val[0] = 0;
      val[1] = SST_CVALG(RHS(1));
      SST_ACLP(LHS, 0);
      SST_SYMP(LHS, getcon(val, DT_INT8));
      SST_DTYPEP(LHS, DT_INT8);
      SST_LSYMP(LHS, 0);
      ast_cnst(top);
    } else {
      SST_LSYMP(LHS, 0);
      SST_DTYPEP(LHS, DT_INT);
      /* value set by scan */
      ast_conval(top);
    }
    break;
  /*
   *      <constant> ::= <int kind const> |
   */
  case CONSTANT2:
    /* token value of <int kind const> is an ST_CONST entry */
    sptr = SST_CVALG(RHS(1));
    dtype = DTYPEG(sptr);
    SST_DTYPEP(LHS, dtype);
    if (dtype == DT_INT8) {
      ast_cnst(top);
    } else {
      SST_CVALP(LHS, CONVAL2G(sptr));
      ast_conval(top);
    }
    break;

  /*
   *      <constant> ::= <real>    |
   */

  case CONSTANT4:
    SST_DTYPEP(LHS, DT_REAL4);
    /* value set by scan */
    ast_conval(top);
    break;
  /*
   *      <constant> ::= <double>  |
   */
  case CONSTANT5:
    SST_DTYPEP(LHS, DT_REAL8);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <quad>     |
   */
  case CONSTANT6:
    SST_DTYPEP(LHS, DT_QUAD);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <complex> |
   */
  case CONSTANT7:
    SST_DTYPEP(LHS, DT_CMPLX8);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <dcomplex> |
   */
  case CONSTANT8:
    SST_DTYPEP(LHS, DT_CMPLX16);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <qcomplex> |
   */
  case CONSTANT9:
    SST_DTYPEP(LHS, DT_QCMPLX);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <nondec const> |
   */
  case CONSTANT10:
    SST_DTYPEP(LHS, DT_WORD);
    /* value set by scan */
    ast_conval(top);
    break;
  /*
   *      <constant> ::= <nonddec const> |
   */
  case CONSTANT11:
    SST_DTYPEP(LHS, DT_DWORD);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <Hollerith>    |
   */
  case CONSTANT12:
    SST_DTYPEP(LHS, DT_HOLL);
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <log const>     |
   */
  case CONSTANT13:
    if (DTY(stb.user.dt_log) == TY_LOG8) {
      if ((INT)SST_CVALG(RHS(1)) == SCFTN_FALSE)
        val[0] = val[1] = 0;
      else if (gbl.ftn_true == -1)
        val[0] = val[1] = -1;
      else {
        val[0] = 0;
        val[1] = 1;
      }
      SST_SYMP(LHS, getcon(val, DT_LOG8));
      SST_DTYPEP(LHS, DT_LOG8);
      ast_cnst(top);
    } else {
      SST_DTYPEP(LHS, DT_LOG);
      /* value set by scan */
      ast_conval(top);
    }
    break;
  /*
   *      <constant> ::= <log kind const> |
   */
  case CONSTANT14:
    /* token value of <log kind const> is an ST_CONST entry */
    sptr = SST_CVALG(RHS(1));
    dtype = DTYPEG(sptr);
    SST_DTYPEP(LHS, dtype);
    if (dtype == DT_LOG8) {
      ast_cnst(top);
      break;
    }
    SST_CVALP(LHS, CONVAL2G(sptr));
    ast_conval(top);
    break;
  /*
   *      <constant> ::= <char literal>
   */
  case CONSTANT15:
    break;

  /*
   *      <constant> ::= <kanji string> |
   */
  case CONSTANT16:
    /*  compute number of Kanji chars in string: */
    sptr = SST_SYMG(RHS(1));         /* ST_CONST/TY_CHAR */
    i = string_length(DTYPEG(sptr)); /* length of string const */
    i = kanji_len((unsigned char *)stb.n_base + CONVAL1G(sptr), i);
    dtype = get_type(2, TY_NCHAR, mk_cval(i, DT_INT4));
    SST_DTYPEP(LHS, dtype);
    val[0] = sptr;
    val[1] = 0;
    SST_CVALP(LHS, getcon(val, dtype));
    ast_cnst(top);
    break;
  /*
   *      <constant> ::= <elp> <expression> <cmplx comma> <expression> )
   */
  case CONSTANT17:
    /*
     * special production to allow complex constants to be formed from
     * "general" real & imag expressions which evaluate to constants.
     * NOTE that for this production, the parser is recovering from
     * '( <expression> , <expression> )'.
     */
    form_cmplx_const(LHS, RHS(2), RHS(4));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<char literal> ::= <quoted string> |
   */
  case CHAR_LITERAL1:
    sptr = SST_SYMG(RHS(1));
    SST_DTYPEP(LHS, DTYPEG(sptr));
    /* value set by scan */
    ast_cnst(top);
    break;
  /*
   *	<char literal> ::= <id> <underscore> <quoted string> |
   */
  case CHAR_LITERAL2:
    sptr = SST_SYMG(RHS(1));
    if (STYPEG(sptr) == ST_PARAM) {
      dtype = DTYPEG(sptr);
      if (!DT_ISINT(dtype))
        error(84, 3, gbl.lineno, SYMNAME(sptr), "- KIND parameter");
      else if (dtype == DT_INT8 || dtype == DT_LOG8) {
        if (get_int_cval(CONVAL1G(sptr)) != 1)
          error(81, 3, gbl.lineno,
                "- KIND parameter has unknown value for quoted string -",
                SYMNAME(sptr));
      } else if (CONVAL1G(sptr) != 1)
        error(81, 3, gbl.lineno,
              "- KIND parameter has unknown value for quoted string -",
              SYMNAME(sptr));
    }
    string_with_kind(top);
    break;
  /*
   *	<char literal> ::= <integer> <underscore> <quoted string>
   */
  case CHAR_LITERAL3:
    dtype = SST_DTYPEG(RHS(1));
    if (dtype == DT_INT8 || dtype == DT_LOG8) {
      if (get_int_cval(SST_SYMG(RHS(1))) != 1)
        error(81, 3, gbl.lineno,
              "- KIND parameter has unknown value for quoted string", CNULL);
    } else if (SST_CVALG(RHS(1)) != 1)
      error(81, 3, gbl.lineno,
            "- KIND parameter has unknown value for quoted string", CNULL);
    string_with_kind(top);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<substring> ::= <char literal> ( <opt sub> : <opt sub> ) |
   */
  case SUBSTRING1:
    SST_DTYPEP(LHS, DTYPEG(SST_SYMG(RHS(1))));
    ast_cnst(top);
    SST_IDP(LHS, S_CONST);
    ch_substring(LHS, RHS(3), RHS(5));
    break;
/*
 *	<substring> ::= <kanji string>  ( <opt sub> : <opt sub> )
 */
  case SUBSTRING2:
    /*  compute number of Kanji chars in string: */
    sptr = SST_SYMG(RHS(1));         /* ST_CONST/TY_CHAR */
    i = string_length(DTYPEG(sptr)); /* length of string const */
    i = kanji_len((unsigned char *)stb.n_base + CONVAL1G(sptr), i);
    dtype = get_type(2, TY_NCHAR, mk_cval(i, DT_INT4));
    SST_DTYPEP(LHS, dtype);
    val[0] = sptr;
    val[1] = 0;
    SST_CVALP(LHS, getcon(val, dtype));
    ast_cnst(top);
    SST_IDP(LHS, S_CONST);
    ch_substring(LHS, RHS(3), RHS(5));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <arith expr> ::= <addition> |
   */
  case ARITH_EXPR1:
    break;
  /*
   *      <arith expr> ::= <term>
   */
  case ARITH_EXPR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <addition> ::= <arith expr> <addop> <term> |
   */
  case ADDITION1:
    goto common_add;
  /*
   *      <addition> ::= <arith expr> <addop> <new term> |
   */
  case ADDITION2:
    if (flg.standard)
      error(170, 2, gbl.lineno, "Unary + or - follows binary + or -", CNULL);
  common_add:
    if (!is_intrinsic_opr(SST_OPTYPEG(RHS(2)), LHS, RHS(1), RHS(3), 0)) {
      binop(LHS, RHS(1), RHS(2), RHS(3));
      if (SST_IDG(LHS) == S_CONST)
        ast = mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS));
      else if (SST_OPTYPEG(RHS(2)) == OP_ADD && !SST_PARENG(RHS(1)))
        ast = reassoc_add(SST_ASTG(RHS(1)), SST_ASTG(RHS(3)), SST_DTYPEG(LHS));
      else
        ast = mk_binop(SST_OPTYPEG(RHS(2)), SST_ASTG(RHS(1)), SST_ASTG(RHS(3)),
                       SST_DTYPEG(LHS));
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    }
    SST_PARENP(LHS, 0);
    break;
  /*
   *      <addition> ::= <new term>
   */
  case ADDITION3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <addop> ::= + |
   */
  case ADDOP1:
    SST_OPTYPEP(LHS, OP_ADD);
    break;
  /*
   *      <addop> ::= -
   */
  case ADDOP2:
    SST_OPTYPEP(LHS, OP_SUB);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <addop list> ::= <addop list> <addop> |
   */
  case ADDOP_LIST1:
    if (SST_OPTYPEG(RHS(1)) == OP_ADD)
      *LHS = *RHS(2);
    else if (SST_OPTYPEG(RHS(2)) == OP_ADD)
      SST_OPTYPEP(LHS, OP_SUB);
    else
      SST_OPTYPEP(LHS, OP_ADD);
    if (flg.standard)
      error(170, 2, gbl.lineno, "Multiple occurrences of unary + or -", CNULL);
    break;
  /*
   *      <addop list> ::= <addop>
   */
  case ADDOP_LIST2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <term> ::= <multiplication> |
   */
  case TERM1:
    break;
  /*
   *      <term> ::= <factor>
   */
  case TERM2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <new term> ::= <addop list> <term>
   */
  case NEW_TERM1:
  unary_arith:
    if (!is_intrinsic_opr(SST_OPTYPEG(RHS(1)), LHS, RHS(2), NULL, 0)) {
      unop(LHS, RHS(1), RHS(2));
      i = SST_OPTYPEG(RHS(1));
      if (SST_IDG(RHS(2)) == S_CONST) {
        if (i == OP_SUB) {
          if (SST_DTYPEG(RHS(2)) == DT_WORD)
            SST_DTYPEP(RHS(2), DT_INT4);
          /* negate constant on semantic stack */
          SST_CVALP(RHS(2),
                    negate_const(SST_CVALG(RHS(2)), (int)SST_DTYPEG(RHS(2))));
          ast = mk_cval1(SST_CVALG(RHS(2)), (int)SST_DTYPEG(RHS(2)));
        } else
          ast = SST_ASTG(RHS(2));
        *LHS = *RHS(2);
      } else {
        mkexpr(RHS(2));
        dtype = SST_DTYPEG(RHS(2));
        if (DTYG(dtype) == TY_STRUCT) {
          error(425, 3, gbl.lineno, NULL, CNULL);
          break;
        }
        SST_DTYPEP(LHS, dtype);
        SST_IDP(LHS, S_EXPR);
        ast = mk_unop(i, SST_ASTG(RHS(2)), SST_DTYPEG(LHS));
      }
      SST_ACLP(LHS, 0);
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    }
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <multiplication> ::= <term> <mult op> <factor> |
   */
  case MULTIPLICATION1:
  /*
   *      <multiplication> ::= <term> <mult op> <new factor>
   */
  case MULTIPLICATION2:
    if (!is_intrinsic_opr(SST_OPTYPEG(RHS(2)), LHS, RHS(1), RHS(3), 0)) {
      binop(LHS, RHS(1), RHS(2), RHS(3));
      if (SST_IDG(LHS) == S_CONST)
        ast = mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS));
      else if (SST_OPTYPEG(RHS(2)) == OP_DIV &&
               DT_ISREAL(DDTG(SST_DTYPEG(LHS))) && !SST_PARENG(RHS(1)) &&
               !SST_PARENG(RHS(3))) {
        ast = combine_fdiv(SST_ASTG(RHS(1)), SST_ASTG(RHS(3)), SST_DTYPEG(LHS));
      } else
        ast = mk_binop(SST_OPTYPEG(RHS(2)), SST_ASTG(RHS(1)), SST_ASTG(RHS(3)),
                       SST_DTYPEG(LHS));
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    }
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <mult op> ::= * |
   */
  case MULT_OP1:
    SST_OPTYPEP(LHS, OP_MUL);
    break;
  /*
   *      <mult op> ::= /
   */
  case MULT_OP2:
    SST_OPTYPEP(LHS, OP_DIV);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <factor> ::= <exponentiation> |
   */
  case FACTOR1:
    break;
  /*
   *      <factor> ::= <primary> |
   */
  case FACTOR2:
    break;
  /*
   *      <factor> ::= <defined unary>
   */
  case FACTOR3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<defined unary> ::= <defined op> <primary>
   */
  case DEFINED_UNARY1:
    defined_operator(SST_SYMG(RHS(1)), LHS, RHS(2), NULL);
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <new factor> ::= <addop list> <factor>
   */
  case NEW_FACTOR1:
    if (flg.standard)
      error(170, 2, gbl.lineno, "Unary + or - follows *, /, or **", CNULL);
    goto unary_arith;

  /* ------------------------------------------------------------------ */
  /*
   *      <exponentiation> ::= <primary> ** <factor>
   */
  case EXPONENTIATION1:
  /*
   *      <exponentiation> ::= <primary> ** <new factor>
   */
  case EXPONENTIATION2:
    if (!is_intrinsic_opr(OP_XTOI, LHS, RHS(1), RHS(3), 0)) {
      SST_OPTYPEP(RHS(2), OP_XTOI);
      binop(LHS, RHS(1), RHS(2), RHS(3));
      if (SST_IDG(LHS) == S_CONST)
        ast = mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS));
      else
        ast =
            mk_binop(OP_XTOI, SST_ASTG(LHS), SST_ASTG(RHS(3)), SST_DTYPEG(LHS));
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    }
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <log expr> ::= <log disjunct> |
   */
  case LOG_EXPR1:
    break;
  /*
   *      <log expr> ::= <eqv or neqv>
   */
  case LOG_EXPR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<eqv or neqv> ::= <log expr> <n eqv op> <log disjunct>
   */
  case EQV_OR_NEQV1:
    common_rel(top, SST_OPTYPEG(RHS(2)), SST_IDG(RHS(2)));
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <log disjunct> ::= <disjunction> |
   */
  case LOG_DISJUNCT1:
    break;
  /*
   *      <log disjunct> ::= <log term>
   */
  case LOG_DISJUNCT2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <disjunction> ::= <OR opr1> .OR. <log term> |
   */
  case DISJUNCTION1:
    common_rel(top, OP_LOR, 0);
    break;
  /*
   *    <disjunction> ::= <OR opr1> .O. <log term>
   */
  case DISJUNCTION2:
    common_rel(top, OP_LOR, TK_ORX);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <OR opr1> ::= <log disjunct>
   */
  case OR_OPR11:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <log term> ::= <conjunction> |
   */
  case LOG_TERM1:
    break;
  /*
   *      <log term> ::= <log factor>
   */
  case LOG_TERM2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <conjunction> ::= <AND opr1> .AND. <log factor>
   */
  case CONJUNCTION1:
    common_rel(top, OP_LAND, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <AND opr1> ::= <log term>
   */
  case AND_OPR11:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <log factor> ::= <log negation> |
   */
  case LOG_FACTOR1:
    break;
  /*
   *      <log factor> ::= <rel operand>  |
   */
  case LOG_FACTOR2:
    break;
  /*
   *      <log factor> ::= <relation>
   */
  case LOG_FACTOR3:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <log negation> ::= .NOT. <log factor>
   */
  case LOG_NEGATION1:
    log_negation(top, 0);
    break;
  /*
   *    <log negation> ::= .N. <log factor>
   */
  case LOG_NEGATION2:
    log_negation(top, TK_NOTX);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <relation> ::= <rel operand> <relop> <rel operand>
   */
  case RELATION1:
    opc = SST_OPTYPEG(RHS(2));
    if (!is_intrinsic_opr(opc, LHS, RHS(1), RHS(3), 0)) {
      SST_OPCP(RHS(2), opc);
      SST_OPTYPEP(RHS(2), OP_CMP);
      binop(LHS, RHS(1), RHS(2), RHS(3));
      if (SST_IDG(LHS) == S_CONST)
        ast = mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS));
      else
        ast =
            mk_binop(opc, SST_ASTG(RHS(1)), SST_ASTG(RHS(3)), SST_DTYPEG(LHS));
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    }
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <rel operand> ::= <primary> |
   */
  case REL_OPERAND1:
    break;
  /*
   *      <rel operand> ::= <exponentiation> |
   */
  case REL_OPERAND2:
    break;
  /*
   *      <rel operand> ::= <multiplication> |
   */
  case REL_OPERAND3:
    break;
  /*
   *      <rel operand> ::= <addition> |
   */
  case REL_OPERAND4:
    break;
  /*
   *      <rel operand> ::= <concatenation>
   */
  case REL_OPERAND5:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <relop> ::= .EQ. |
   */
  case RELOP1:
    SST_OPTYPEP(LHS, OP_EQ);
    break;
  /*
   *      <relop> ::= .GE. |
   */
  case RELOP2:
    SST_OPTYPEP(LHS, OP_GE);
    break;
  /*
   *      <relop> ::= .GT. |
   */
  case RELOP3:
    SST_OPTYPEP(LHS, OP_GT);
    break;
  /*
   *      <relop> ::= .LE. |
   */
  case RELOP4:
    SST_OPTYPEP(LHS, OP_LE);
    break;
  /*
   *      <relop> ::= .LT. |
   */
  case RELOP5:
    SST_OPTYPEP(LHS, OP_LT);
    break;
  /*
   *      <relop> ::= .NE.
   */
  case RELOP6:
    SST_OPTYPEP(LHS, OP_NE);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <char expr> ::= <arith expr> |
   */
  case CHAR_EXPR1:
    break;
  /*
   *      <char expr> ::= <concatenation>
   */
  case CHAR_EXPR2:
    break;

  /* ------------------------------------------------------------------ */
  /*
   *      <concatenation> ::= <char expr> '//' <arith expr>
   */
  case CONCATENATION1:
    if (!is_intrinsic_opr(OP_CAT, LHS, RHS(1), RHS(3), 0)) {
      SST_OPTYPEP(RHS(2), OP_CAT);
      binop(LHS, RHS(1), RHS(2), RHS(3));
      if (SST_IDG(LHS) == S_CONST)
        ast = mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS));
      else
        ast = mk_binop(OP_CAT, SST_ASTG(RHS(1)), SST_ASTG(RHS(3)),
                       SST_DTYPEG(LHS));
      SST_ASTP(LHS, ast);
      SST_SHAPEP(LHS, A_SHAPEG(ast));
    }
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  /*
   *	<defined binary> ::= <expression> <defined op> <log expr>
   */
  case DEFINED_BINARY1:
    defined_operator(SST_SYMG(RHS(2)), LHS, RHS(1), RHS(3));
    SST_PARENP(LHS, 0);
    break;

  /* ------------------------------------------------------------------ */
  default:
    interr("semant2:bad rednum", rednum, 3);
  }
}

static void
common_rel(SST *top, int opc, int tkn_alias)
{
  if (!is_intrinsic_opr(opc, LHS, RHS(1), RHS(3), tkn_alias)) {
    int ast;
    if (opc == OP_LAND || opc == OP_LOR) {
      /* Left operand is short circuitable.  The left operand of a
       * logical expression needs evaluating immediately to avoid
       * having the second operand evaluated before the first.
       * Short circuiting is from left to right.  That is, if an
       * operand on the left determines that the operand(s) on the
       * right do not need evaluating then their evaluation is
       * avoided.
       */
      chklog(RHS(1));
    }
    SST_OPTYPEP(RHS(2), OP_LOG);
    SST_OPCP(RHS(2), opc);
    binop(LHS, RHS(1), RHS(2), RHS(3));
    if (SST_IDG(LHS) == S_CONST)
      ast = mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS));
    else if (SST_ASTG(RHS(3)) == 0)
      /* short circuit optimization occurred and the result is
       * an expression
       */
      ast = SST_ASTG(LHS);
    else
      ast = mk_binop(opc, SST_ASTG(RHS(1)), SST_ASTG(RHS(3)), SST_DTYPEG(LHS));
    SST_ASTP(LHS, ast);
    SST_SHAPEP(LHS, A_SHAPEG(ast));
  }
  SST_PARENP(LHS, 0);
}

static void
log_negation(SST *top, int tkn_alias)
{
  if (!is_intrinsic_opr(OP_LNOT, LHS, RHS(2), NULL, tkn_alias)) {
    int ast;
    if (SST_IDG(RHS(2)) == S_CONST) {
      int dtype;
      if (SST_ISNONDECC(RHS(2)) || (SST_DTYPEG(RHS(2)) == DT_DWORD) ||
          SST_DTYPEG(RHS(2)) == DT_INT8)
        cngtyp(RHS(2), DT_LOG);
      dtype = SST_DTYPEG(RHS(2));
      if (dtype == DT_LOG8) {
        int sptr = SST_CVALG(RHS(2));
        INT val[2];
        if (CONVAL2G(sptr))
          /* constant is true */
          val[0] = val[1] = 0;
        else if (gbl.ftn_true == -1)
          val[0] = val[1] = -1;
        else {
          val[0] = 0;
          val[1] = 1;
        }
        sptr = getcon(val, DT_LOG8);
        SST_DTYPEP(LHS, DT_LOG8);
        SST_CVALP(LHS, sptr);
        ast = mk_cval1(sptr, DT_LOG8);
      } else {
        SST_CVALP(LHS, SCFTN_NEGATE(SST_CVALG(RHS(2))));
        SST_DTYPEP(LHS, DT_LOG4);
        ast = mk_cval1(SST_CVALG(LHS), DT_LOG4);
      }
      SST_IDP(LHS, S_CONST);
    } else {
      SST_IDP(LHS, S_EXPR);
      chklog(RHS(2));
      SST_DTYPEP(LHS, SST_DTYPEG(RHS(2)));
      ast = mk_unop(OP_LNOT, SST_ASTG(RHS(2)), SST_DTYPEG(LHS));
    }
    SST_ASTP(LHS, ast);
    SST_SHAPEP(LHS, A_SHAPEG(ast));
    SST_ACLP(LHS, 0); /* prevent UMR */
  }
  SST_PARENP(LHS, 0);
}

static void
var_ref_list(SST *top, int rhstop)
{
  ITEM *itemp;
  SST *e1 = (SST *)getitem(0, sizeof(SST));
  *e1 = *RHS(rhstop);
  itemp = mkitem(e1);
  if (rhstop == 1)
    SST_BEGP(LHS, itemp);
  else
    (SST_ENDG(RHS(1)))->next = itemp;
  SST_ENDP(LHS, itemp);
}

static void
ssa_list(SST *top, int rhstop)
{
  SST *e1 = SST_E1G(RHS(rhstop));
  if (sem.in_struct_constr) {
    ACL *aclp;
    if (SST_IDG(e1) == S_ACONST || SST_IDG(e1) == S_SCONST) {
      aclp = SST_ACLG(e1);
    } else {
      /* put in ACL */
      aclp = GET_ACL(15);
      aclp->id = AC_EXPR;
      aclp->repeatc = aclp->size = 0;
      aclp->next = NULL;
      aclp->subc = NULL;
      aclp->u1.stkp = e1;
    }
    if (rhstop == 1) {
      SST_BEGP(LHS, (ITEM *)aclp);
    } else
      ((ACL *)SST_ENDG(RHS(1)))->next = aclp;
    SST_ENDP(LHS, (ITEM *)aclp);

  } else {
    /* put in ITEM */
    ITEM *itemp = mkitem(e1);
    if (rhstop == 1)
      SST_BEGP(LHS, itemp);
    else
      (SST_ENDG(RHS(1)))->next = itemp;
    SST_ENDP(LHS, itemp);
  }
}

static void
ast_cnst(SST *top)
{
  SST_ASTP(LHS, mk_cnst(SST_CVALG(LHS)));
  SST_SHAPEP(LHS, 0);
}

static void
ast_conval(SST *top)
{
  SST_ACLP(LHS, 0); /* prevent UMR */
  SST_ASTP(LHS, mk_cval1(SST_CVALG(LHS), (int)SST_DTYPEG(LHS)));
  SST_SHAPEP(LHS, 0);
}

static void
string_with_kind(SST *top)
{
  SST_DTYPEP(LHS, DTYPEG(SST_SYMG(RHS(3))));
  SST_CVALP(LHS, SST_SYMG(RHS(3)));
  /* value set by scan */
  ast_cnst(top);
}

static int
combine_fdiv(int l_ast, int r_ast, int dt)
{
  if (flg.opt != 0 && XBIT(15, 0x2) && A_TYPEG(l_ast) == A_BINOP &&
      A_OPTYPEG(l_ast) == OP_DIV) {
    int l, r;

    l = A_LOPG(l_ast);
    r = A_ROPG(l_ast);
    r = mk_binop(OP_MUL, r, r_ast, dt);
    l = mk_binop(OP_DIV, l, r, dt);
    return l;
  }
  return mk_binop(OP_DIV, l_ast, r_ast, dt);
}

static int
has_private(int in_dtype)
{
  int tag;
  int mem_sptr;
  int dtype;
  int prv = 0;

  dtype = in_dtype;
  if (DTY(dtype) == TY_ARRAY)
    dtype = DTY(dtype + 1);

  tag = DTY(dtype + 3);
  if (VISITG(tag))
    return 0;
  if (PRIVATEG(tag) && test_private_dtype(dtype))
    return 1;
  VISITP(tag, 1);

  mem_sptr = DTY(dtype + 1);
  for (; mem_sptr != NOSYM; mem_sptr = SYMLKG(mem_sptr)) {
    if (is_iso_cptr(DTYPEG(mem_sptr))) {
      continue;
    }
    if (is_tbp_or_final(mem_sptr)) {
      /* skip tbp */
      continue;
    }
    if (PRIVATEG(mem_sptr)) {
      prv = 1;
      break;
    }
    if (DTYG(DTYPEG(mem_sptr)) == TY_DERIVED)
      if (has_private(DTYPEG(mem_sptr))) {
        prv = 1;
        break;
      }
  }

  VISITP(tag, 0);
  return prv;
} /* has_private */

/**
   \brief Check whether the dtype is inside a valid scope.
 */
int
test_private_dtype(int dtype)
{
  /* get the tag of the derived type */
  int tag;
  tag = DTY(dtype + 3);
  if (tag) {
    SPTR tagscope, scope, prev_scope;
    tagscope = SCOPEG(tag);
    for (scope = stb.curr_scope, prev_scope = NOSYM; scope > NOSYM;
         prev_scope = scope, scope = SCOPEG(scope)) {

      if (scope == prev_scope)
        scope = 0;

      if (tagscope == scope)
        break;

      if (scope > NOSYM && STYPEG(scope) == ST_MODULE && 
          ANCESTORG(scope) && tagscope == ANCESTORG(scope))
        break;
    }
    if (tagscope && scope <= NOSYM) {
      return 1;
    }
  }
  return 0;
} /* test_private_dtype */

static void
rewrite_cmplxpart_rval(SST *e)
{
  int ast;
  ITEM *list;
  const char *intrnm;
  SST *arg;
  int sptr;
  int part; /* 1==> real, 2==>imag */
  int dtype;

  if ((ast = SST_ASTG(e)) && A_TYPEG(ast) == A_MEM &&
      DT_ISCMPLX(A_DTYPEG(A_PARENTG(ast))) && A_TYPEG(A_MEMG(ast)) == A_ID) {
    sptr = A_SPTRG(A_MEMG(ast));
    if (strcmp(SYMNAME(sptr), "re") == 0) {
      part = 1;
    } else if (strcmp(SYMNAME(sptr), "im") == 0) {
      part = 2;
    } else {
      return;
    }
    dtype = A_DTYPEG(ast);
    switch (DTY(dtype)) {
    case TY_REAL:
      intrnm = part == 1 ? "real" : "imag";
      break;
    case TY_DBLE:
      intrnm = part == 1 ? "dreal" : "dimag";
      break;
#ifdef TARGET_SUPPORTS_QUADFP
    case TY_QUAD:
      intrnm = part == 1 ? "qreal" : "qimag";
      break;
#endif
    default:
      interr("rewrite_cmplxpart_rval: unexpected type", DTY(dtype), 3);
    }
    sptr = getsymbol(intrnm);
    if (IS_INTRINSIC(STYPEG(sptr))) {
      setimplicit(sptr);
    }

    SST_IDP(e, S_IDENT);
    SST_SYMP(e, sptr);
    /*SST_DTYPEP(e, dtype);*/

    arg = (SST *)getitem(0, sizeof(SST));
    BZERO(arg, SST, 1);
    SST_IDP(arg, S_EXPR);
    SST_ASTP(arg, A_PARENTG(ast));
    SST_DTYPEP(arg, SST_DTYPEG(e));
    SST_PARENP(arg, 0);

    list = mkitem(arg);
    list->ast = 0;
    mkvarref(e, list);
  }
}

static void
form_cmplx_const(SST *res, SST *rp, SST *ip)
{
  int dtype, cdtype;
  int r, i;
  int ast;
  INT val[2];

  if (SST_IDG(rp) != S_CONST || SST_IDG(ip) != S_CONST) {
    dtype = cdtype = DT_CMPLX;
    val[0] = 0;
    val[1] = 0;
    errsev(87);
  } else {
    r = SST_DTYPEG(rp);
    i = SST_DTYPEG(ip);
#ifdef TARGET_SUPPORTS_QUADFP
    if (r == DT_QUAD || r == DT_QCMPLX ||
        i == DT_QUAD || i == DT_QCMPLX) {
      dtype = DT_QUAD;
      cdtype = DT_QCMPLX;
    } else
#endif
    if (r == DT_DBLE || r == DT_DCMPLX ||
        i == DT_DBLE || i == DT_DCMPLX) {
      dtype = DT_DBLE;
      cdtype = DT_DCMPLX;
    } else {
      dtype = DT_REAL;
      cdtype = DT_CMPLX;
    }

    cngtyp(rp, dtype);
    val[0] = SST_CVALG(rp);
    cngtyp(ip, dtype);
    val[1] = SST_CVALG(ip);
  }
  SST_IDP(res, S_CONST);
  SST_DTYPEP(res, cdtype);
  SST_CVALP(res, getcon(val, cdtype));
  ast = mk_cnst(SST_CVALG(res));
  SST_ASTP(res, ast);
  SST_SHAPEP(res, 0);

}

static int
_assoc_term(int op)
{
  if (!XBIT(19, 0x1000000) || A_ISLVAL(A_TYPEG(op)) || A_ALIASG(op))
    return 1;
  return 0;
}

static int
reassoc_add(int lop, int rop, int dtype)
{
  int ast;
  int op1, op2;

  if (XBIT(19, 0x20000) || !_assoc_term(rop) || !DT_ISREAL(DDTG(dtype)))
    return mk_binop(OP_ADD, lop, rop, dtype);

  if (A_TYPEG(lop) != A_BINOP || A_OPTYPEG(lop) != OP_ADD)
    return mk_binop(OP_ADD, lop, rop, dtype);

  op1 = A_LOPG(lop);
  op2 = A_ROPG(lop);
  if (A_TYPEG(op1) == A_BINOP) {
    if (A_OPTYPEG(op1) != OP_ADD || !_assoc_term(op2))
      return mk_binop(OP_ADD, lop, rop, dtype);
    op2 = mk_binop(OP_ADD, op2, rop, dtype);
    ast = mk_binop(OP_ADD, op1, op2, dtype);
  } else if (_assoc_term(op1) && A_TYPEG(op2) == A_BINOP &&
             A_OPTYPEG(op2) == OP_ADD) {
    op1 = mk_binop(OP_ADD, op1, rop, dtype);
    ast = mk_binop(OP_ADD, op2, op1, dtype);
  } else
    ast = mk_binop(OP_ADD, lop, rop, dtype);
  return ast;
}

static int
get_mem_sptr_by_name(char *name, int dtype)
{

  int mem, sptr;

  if (DTY(dtype) != TY_DERIVED)
    return 0;
  for (mem = DTY(dtype + 1); mem > NOSYM; mem = SYMLKG(mem)) {
#ifdef PARENTG
    if (PARENTG(mem)) {
      sptr = get_mem_sptr_by_name(name, DTYPEG(PARENTG(mem)));
      if (sptr)
        return sptr;
    }
#endif
    if (strcmp(name, SYMNAME(mem)) == 0)
      return mem;
  }
  return 0;
}

static ITEM *
mkitem(SST *stkp)
{
  ITEM *item = (ITEM *)getitem(0, sizeof(ITEM));
  item->next = ITEM_END;
  item->t.stkp = stkp;
  return item;
}

/** \brief Return std range to mark the stds generated by loop body or loop
           control in an implied-do loop.
    \param prev the STD_LAST when the semantic analysis of
                an implied-do loop is started.
    \param mid  the STD_LAST when meeting the <idc eq>.
    \param end  the STD_LAST when the semantic analysis of
                an implied-do loop is finished.
 */
static STD_RANGE *
mk_ido_std_range(int prev, int mid, int end)
{
  STD_RANGE *range = NULL;
  if (prev == end)
    return NULL;
  range = (STD_RANGE *)getitem(0, sizeof(STD_RANGE));
  range->start = prev == mid ? 0 : STD_NEXT(prev);
  range->mid = mid;
  range->end = end;
  return range;
}
