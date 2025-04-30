/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 *  \brief Utility routines used by semantic analyzer.
 *
 *  Fortran front-end utility routines used by Semantic Analyzer to process
 *  functions, subroutines, predeclareds, statement functions, etc.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "scan.h"
#include "semstk.h"
#include "pd.h"
#include "machar.h"
#include "ast.h"
#include "rte.h"
#include "rtlRtns.h"

static LOGICAL get_keyword_args(ITEM *, int, const char *, int, int);
static int get_fval_array(int);
static LOGICAL cmpat_arr_arg(int, int);
static void dump_stfunc(int);

/*---------------------------------------------------------------------*/

/*
    routines to define statement functions and reference statement
    functions
*/

/*  define structures needed for statement function processing: */

/** \brief Information about statement function arguments. */
typedef struct _arginfo {
  int dtype;             /**< data type of dummy argument  */
  int formal;            /**< dummy's ast */
  int actual;            /**< ast of argument to replace corresponding dummy */
  int save;              /**< save/restore field */
  unsigned refd : 1;     /**< if set, formal is referenced */
  struct _arginfo *next; /**< next argument info record */
} ARGINFO;

/** \brief Statement function used in the right-hand side. */
typedef struct _sfuse {
  int node;            /**< unique id ast to be replaced when invoked */
  struct _sfdsc *dsc;  /**< its statement function descriptor */
  ARGINFO *args;       /**< arguments to the statement function */
  struct _sfuse *next; /**< next statement function used */
} SFUSE;

/** \brief Statement function descriptor. */
typedef struct _sfdsc {
  int dtype;     /**< dtype of the statement function */
  int rhs;       /**< ast of right hand side */
  ARGINFO *args; /**< ptr to list of arginfo records */
  SFUSE *l_use;  /**< list of statement functions used in rhs */
} SFDSC;

/** \brief List of statement functions used in a definition.
 *
 *  Keep track of the statement functions used in a definition and the
 *  arguments to those statement functions which need to be evaluated
 *  and saved in a temporary. The order of the list is in 'evaluation' order.
 */
static SFUSE *l_sfuse = NULL;
static SFUSE *end_sfuse = NULL; /**< End of list -- always add here */

static void asn_sfuse(SFUSE *);

static void _non_private(int, int *);
static ITEM *stfunc_argl;

static int pass_position = 0;
static int pass_object_dummy = 0;

/** \brief Set position of type-bound procedure passed object location. */
void
set_pass_objects(int pos, int pod)
{
  pass_position = pos;
  pass_object_dummy = pod;
}

/** \brief Perform error checking and create statement function descriptor
 *         for statement function definition.
 */
int
define_stfunc(int sptr, ITEM *argl, SST *estk)
{
  int expr;
  int arg, dtype;
  ITEM *itemp;
  ARGINFO *arginfo, *lastarg;
  SFDSC *sfdsc;
  SST *stkptr;
  int ast;
  static int last_stfunc; /* last ST_FUNC created */
  SFUSE *sfuse;

  expr = mkexpr(estk);
  ast = SST_ASTG(estk);
  stfunc_argl = argl;

  ast_visit(1, 1);                             /* marks ast#1 visited */
  ast_traverse(ast, NULL, _non_private, NULL); /* vist each ast in the rhs */

  /* traverse the args to any statement functions ref'd in this one */
  for (sfuse = l_sfuse; sfuse; sfuse = sfuse->next)
    for (arginfo = sfuse->args; arginfo; arginfo = arginfo->next)
      ast_traverse(arginfo->actual, NULL, NULL, NULL);

  /*  allocate and initialize statement function descriptor  */

  /* NOTE: 9/17/97, area 8 is used for stmt functions -- need to keep
   * just in case the defs appear in a containing subprogram.
   */
  sfdsc = (SFDSC *)getitem(8, sizeof(SFDSC));
  sfdsc->args = NULL;

  /*  scan list of dummy arguments and process each argument; all arguments
   *  have been validated by semant.
   */
  lastarg = NULL;
  for (itemp = argl; itemp != ITEM_END; itemp = itemp->next) {
    int old, new;
    stkptr = itemp->t.stkp;
    arg = SST_SYMG(stkptr);
    if (ARGINFOG(arg)) { /* duplicate dummy */
      error(42, 3, gbl.lineno, SYMNAME(arg), CNULL);
      ast_unvisit();
      return 0;
    }
    dtype = DTYPEG(arg);
    /*
     * allocate and initialize an arginfo record for this dummy,
     * and link it into end of argument list
     */
    arginfo = (ARGINFO *)getitem(8, sizeof(ARGINFO));
    old = SST_ASTG(itemp->t.stkp);
    /*
     * replace the ast of the formal argument with a unique ast; can't
     * share asts of any formals with any nested statement functions.
     */
    new = new_node(A_ID);
    A_SPTRP(new, arg);
    A_DTYPEP(new, dtype);
    ast_replace(old, new);
    arginfo->formal = new;
    arginfo->dtype = dtype;
    arginfo->next = NULL;
    arginfo->refd = A_VISITG(old) != 0;
    if (lastarg == NULL) /* this is first argument */
      sfdsc->args = arginfo;
    else
      lastarg->next = arginfo;
    lastarg = arginfo;
    ARGINFOP(arg, put_getitem_p(arginfo));
  }
  /*
   * rewrite the rhs of the statement function and the actual argument
   * asts of any statement functions ref'd in this one; this replaces
   * the original asts of the formal arguments with their new asts.
   */
  ast = ast_rewrite(ast);
  for (sfuse = l_sfuse; sfuse; sfuse = sfuse->next)
    for (arginfo = sfuse->args; arginfo; arginfo = arginfo->next)
      arginfo->actual = ast_rewrite(arginfo->actual);

  ast_unvisit();

  sfdsc->rhs = ast;
  sfdsc->l_use = l_sfuse; /* list of statement functions used */
  end_sfuse = l_sfuse = NULL;

  /*  set ARGINFO fields of dummies back to 0  */

  for (itemp = argl; itemp != ITEM_END; itemp = itemp->next)
    ARGINFOP(SST_SYMG(itemp->t.stkp), 0);

  SFDSCP(sptr, put_getitem_p(sfdsc));
  STYPEP(sptr, ST_STFUNC);
  SFASTP(sptr, ast);
  sfdsc->dtype = DTYPEG(sptr);
  if (gbl.stfuncs == NOSYM)
    gbl.stfuncs = sptr;
  else
    SYMLKP(last_stfunc, sptr);
  last_stfunc = sptr;

  if (DBGBIT(3, 16))
    dump_stfunc(sptr);

  return ast;
}

/** \brief AST visitor function
 *
 *  This is passed to ast_traverse() to add variables with add_non_private.
 */
static void
_non_private(int ast, int *dummy)
{
  int sptr;
  if (!flg.smp)
    return;
  if (A_TYPEG(ast) != A_ID)
    return;
  sptr = A_SPTRG(ast);
  if (ST_ISVAR(STYPEG(sptr))) {
    /*
     * Make sure that sptr is not the dummy arg to the statement function.
     */
    ITEM *itemp;
    for (itemp = stfunc_argl; itemp != ITEM_END; itemp = itemp->next) {
      SST *stkptr;
      int arg;
      stkptr = itemp->t.stkp;
      arg = SST_SYMG(stkptr);
      if (arg == sptr)
        break;
    }
    if (itemp == ITEM_END)
      add_non_private(sptr);
  }
}

/*---------------------------------------------------------------------*/

/** \brief Write out statement function descriptor to debug file. */
static void
dump_stfunc(int sptr)
{
  SFDSC *sfdsc;
  ARGINFO *arginfo;
  SFUSE *sfuse;

  sfdsc = (SFDSC *)get_getitem_p(SFDSCG(sptr));
  fprintf(gbl.dbgfil, "\nSTATEMENT FUNCTION DEFN: %s, sfdsc: %p, dtype: %d\n",
          SYMNAME(sptr), (void *)sfdsc, sfdsc->dtype);

  for (arginfo = sfdsc->args; arginfo; arginfo = arginfo->next) {
    fprintf(gbl.dbgfil, "    arg: %p   ast: %d   dtype: %d   refd: %d\n",
            (void *)arginfo, arginfo->formal, arginfo->dtype, arginfo->refd);
    dump_one_ast(arginfo->formal);
  }
  fprintf(gbl.dbgfil, "\nRHS:");
  dump_ast_tree(sfdsc->rhs);
  fprintf(gbl.dbgfil, "\n");
  fprintf(gbl.dbgfil, "sfuse:\n");
  for (sfuse = sfdsc->l_use; sfuse; sfuse = sfuse->next) {
    fprintf(gbl.dbgfil, "<sfdsc %p, exprs: %p>\n", (void *)sfuse->dsc,
            (void *)sfuse->args);
    for (arginfo = sfuse->args; arginfo; arginfo = arginfo->next) {
      fprintf(gbl.dbgfil, "    arginfo: %p  actual: %d   dtype: %d\n",
              (void *)arginfo, arginfo->actual, arginfo->dtype);
      dump_one_ast(arginfo->actual);
    }
  }
  fprintf(gbl.dbgfil, "\n");
}

/*---------------------------------------------------------------------*/

int
ref_stfunc(SST *stktop, ITEM *args)
{
  int sptr;
  int dtype;
  ITEM *itemp;
  SFDSC *sfdsc;
  ARGINFO *arginfo;
  SFUSE *sfuse;
  ARGINFO *ai;
  int ast;
  int tmp;
  int new;
  int asn;

  sptr = SST_SYMG(stktop);
  if (DBGBIT(3, 16))
    fprintf(gbl.dbgfil, "\nInvoking statement function %s\n", SYMNAME(sptr));
  dtype = DTYPEG(sptr);
  sfdsc = (SFDSC *)get_getitem_p(SFDSCG(sptr));
  if (sem.in_stfunc) {
    /* NOTE: 9/17/97, area 8 is used for stmt functions -- need to keep
     * just in case the defs appear in a containing subprogram.
     */
    sfuse = (SFUSE *)getitem(8, sizeof(SFUSE));
    sfuse->dsc = sfdsc;
    sfuse->args = NULL;
    /*
     * create a unique id AST whose sptr is the statement function
     * which is referenced; this id will be replaced by the statement
     * function's right-hand side (after argument substitution).
     */
    sfuse->node = new_node(A_ID);
    A_SPTRP(sfuse->node, sptr);
    A_DTYPEP(sfuse->node, dtype);
    /*
     * add this statement function to the 'global' statement function
     * use; when the definition ultimately occurs, the pointer to the
     * list will be stored in the descriptor of the statement which
     * is defined.
     */
    if (end_sfuse == NULL)
      end_sfuse = l_sfuse = sfuse;
    else
      end_sfuse->next = sfuse;
    end_sfuse = sfuse;
    sfuse->next = NULL;
    if (DBGBIT(3, 16))
      fprintf(gbl.dbgfil, "%s in statement fcn def, use %p, node %d\n",
              SYMNAME(sptr), (void *)sfuse, sfuse->node);
  }
  /*
   * scan thru actual argument list, and list of dummy arg info
   * records in parallel to check type and create asts for actual args
   */
  for (itemp = args, arginfo = sfdsc->args;
       itemp != ITEM_END && arginfo != NULL;
       itemp = itemp->next, arginfo = arginfo->next) {
    if (SST_IDG(itemp->t.stkp) == S_KEYWORD) {
      error(79, 3, gbl.lineno, scn.id.name + SST_CVALG(itemp->t.stkp), CNULL);
      itemp->t.stkp = SST_E3G(itemp->t.stkp);
      arginfo->refd = 0;
    } else if (SST_IDG(itemp->t.stkp) == S_TRIPLE ||
               SST_IDG(itemp->t.stkp) == S_STAR) {
      error(155, 3, gbl.lineno,
            "An argument to this statement function looks "
            "like an array section subscript",
            CNULL);
      continue;
    }

    if (arginfo->refd)
      (void)chktyp(itemp->t.stkp, arginfo->dtype, TRUE);
    else /* although arg isn't refd, ensure ast for the actual exists */
      (void)mkexpr(itemp->t.stkp);
    ast = SST_ASTG(itemp->t.stkp);
    arginfo->actual = ast;
  }

  /*  check that number of arguments is correct  */

  if (itemp != ITEM_END || arginfo != NULL)
    error(85, 3, gbl.lineno, SYMNAME(sptr), CNULL);
  /*
   * If in a statement function definition, create a list of actual
   * arguments passed to the statement function.
   */
  if (sem.in_stfunc) {
    for (arginfo = sfdsc->args; arginfo != NULL; arginfo = arginfo->next) {
      ai = (ARGINFO *)getitem(8, sizeof(ARGINFO));
      ai->next = sfuse->args;
      sfuse->args = ai;
      ai->actual = arginfo->actual;
      ai->formal = arginfo->formal;
      ai->dtype = arginfo->dtype;
      ai->refd = arginfo->refd;
      if (DBGBIT(3, 16)) {
        fprintf(gbl.dbgfil, "expr to be substituted, ast %d, arginfo %p\n",
                ai->actual, (void *)ai);
        fprintf(gbl.dbgfil, "formal(%d):\n", ai->formal);
        dbg_print_ast(ai->formal, gbl.dbgfil);
        fprintf(gbl.dbgfil, "actual(%d):\n", ai->actual);
        dbg_print_ast(ai->actual, gbl.dbgfil);
      }
    }
    ast = sfuse->node;
    goto return_it;
  }
  /*
   * replace uses of the dummy arguments with the actual arguments.
   */
  ast_visit(1, 1);
  for (arginfo = sfdsc->args; arginfo != NULL; arginfo = arginfo->next) {
    if (!arginfo->refd) {
      if (DBGBIT(3, 16)) {
        fprintf(gbl.dbgfil, "\n   skipping unref'd arg");
        dump_ast_tree(arginfo->formal);
      }
      continue;
    }
    ast = arginfo->actual;
    if (A_CALLFGG(ast) || A_TYPEG(ast) == A_CONV) {
      /*
       * evaluate and assign  the argument to a temporary if:
       * 1.  argument contains a function call - ensure the
       *     the function is evaluated just once, or
       * 2.  the argument is converted to another type - need to
       *     preserve the type.
       */
      tmp = get_temp(arginfo->dtype);
      if (DBGBIT(3, 16))
        fprintf(gbl.dbgfil, "\n   create temp.1 %s\n", SYMNAME(tmp));
      new = mk_id(tmp);
      asn = mk_assn_stmt(new, ast, arginfo->dtype);
      (void)add_stmt(asn);
      arginfo->actual = new;
    }
    ast_replace(arginfo->formal, arginfo->actual);
    if (DBGBIT(3, 16)) {
      fprintf(gbl.dbgfil, "\n   replace %d:\n", arginfo->formal);
      dbg_print_ast(arginfo->formal, gbl.dbgfil);
      fprintf(gbl.dbgfil, "   with %d:\n", arginfo->actual);
      /*dump_ast_tree(arginfo->actual);*/
      dbg_print_ast(arginfo->actual, gbl.dbgfil);
    }
  }

  /* evaluate any statement functions which appeared in the definition
   * of the statement function.
   */
  asn_sfuse(sfdsc->l_use);

  ast = ast_rewrite(sfdsc->rhs);
  ast_unvisit();
  if (DBGBIT(3, 16)) {
    fprintf(gbl.dbgfil, "\n   statement function result %d\n", ast);
    /*dump_ast_tree(ast);*/
    dbg_print_ast(ast, gbl.dbgfil);
    fprintf(gbl.dbgfil, "\n");
  }
  if (!sem.in_stfunc && A_TYPEG(ast) == A_CONV) {
    /* TBD:  replace with an expression to convert the type.
     * For now, just
     * assign the result to a temporary if the result is converted
     * to another type - need to preserve the type.
     */
    tmp = get_temp(dtype);
    if (DBGBIT(3, 16))
      fprintf(gbl.dbgfil, "\n   create temp.2 %s\n", SYMNAME(tmp));
    new = mk_id(tmp);
    asn = mk_assn_stmt(new, ast, dtype);
    (void)add_stmt(asn);
    ast = new;
  }
  if (gbl.internal > 1) {
    ast_visit(1, 1); /* marks ast#1 visited */
    ast_traverse(ast, NULL, set_internref_stfunc,
                 NULL); /* vist each ast in the rhs */
    ast_unvisit();
  }

return_it:
  SST_ASTP(stktop, ast);
  SST_SHAPEP(stktop, 0);
  SST_DTYPEP(stktop, dtype);
  SST_IDP(stktop, S_EXPR);

  return 1;
}

static void
asn_sfuse(SFUSE *l_use)
{
  SFUSE *sfuse;
  ARGINFO *expr;
  int ast;
  int tmp;
  int asn;
  int new;

  if (DBGBIT(3, 16))
    fprintf(gbl.dbgfil, "asn_sfuse entered\n");
  for (sfuse = l_use; sfuse != NULL; sfuse = sfuse->next) {
    if (DBGBIT(3, 16))
      fprintf(gbl.dbgfil, "\n    asn_sfuse, begin sfuse %p, node %d\n",
              (void *)sfuse, sfuse->node);

    /* substitute the actual arguments for the corresponding formals */

    for (expr = sfuse->args; expr != NULL; expr = expr->next) {
      expr->save = A_REPLG(expr->formal);
      if (!expr->refd) {
        if (DBGBIT(3, 16)) {
          fprintf(gbl.dbgfil, "\n   asn_sfuse: skipping unref'd arg");
          dump_ast_tree(expr->formal);
        }
        continue;
      }
      ast = ast_rewrite(expr->actual);
      if (A_CALLFGG(ast) || A_TYPEG(ast) == A_CONV) {
        tmp = get_temp(expr->dtype);
        new = mk_id(tmp);
        asn = mk_assn_stmt(new, ast, expr->dtype);
        (void)add_stmt(asn);
        ast = new;
      }
      ast_replace(expr->formal, ast);
      if (DBGBIT(3, 16)) {
        fprintf(gbl.dbgfil, "    asn_sfuse, replace formal %d\n", expr->formal);
        dbg_print_ast(expr->formal, gbl.dbgfil);
        fprintf(gbl.dbgfil, "    asn_sfuse, with %d\n", ast);
        dbg_print_ast(ast, gbl.dbgfil);
      }
    }
    /*
     * evaluate any statement functions which were invoked in this
     * statement function.
     */
    asn_sfuse(sfuse->dsc->l_use);
    /*
     * replace the statement function with the evaluation of its
     * right-hand side.
     */
    ast = ast_rewrite(sfuse->dsc->rhs);
    if (DBGBIT(3, 16)) {
      fprintf(gbl.dbgfil, "    asn_sfuse, rewrite %d:\n", sfuse->dsc->rhs);
      dbg_print_ast(sfuse->dsc->rhs, gbl.dbgfil);
      fprintf(gbl.dbgfil, "    asn_sfuse, as %d:\n", ast);
      dbg_print_ast(ast, gbl.dbgfil);
    }
    if (A_CALLFGG(ast) || A_TYPEG(ast) == A_CONV) {
      tmp = get_temp(sfuse->dsc->dtype);
      if (DBGBIT(3, 16))
        fprintf(gbl.dbgfil, "    asn_sfuse, create temp %s\n", SYMNAME(tmp));
      new = mk_id(tmp);
      asn = mk_assn_stmt(new, ast, sfuse->dsc->dtype);
      (void)add_stmt(asn);
      ast = new;
    }
    ast_replace(sfuse->node, ast);
    if (DBGBIT(3, 16))
      fprintf(gbl.dbgfil, "    asn_sfuse, end sfuse %p, node %d\n",
              (void *)sfuse, A_REPLG(sfuse->node));
    /*
     * cleanup in this order: zero out the REPL fields of the right-hand
     * side and restore the state of the formals to the statement function.
     */
    ast_clear_repl(sfuse->dsc->rhs);
    for (expr = sfuse->args; expr != NULL; expr = expr->next)
      A_REPLP(expr->formal, expr->save);
  }
  if (DBGBIT(3, 16))
    fprintf(gbl.dbgfil, "asn_sfuse returned\n");
}

/*---------------------------------------------------------------------*/

/** \brief Check and write ILMs for a subprogram argument.
 *  \param stkptr a stack entry representing a subprogram argument
 *  \param dtype  used to pass out data type of argument
 *  \return       sptr for alternate return label
 */
int
mkarg(SST *stkptr, int *dtype)
{
  int sptr, sp2, ast;
  int dt;

again:
  switch (SST_IDG(stkptr)) {
  case S_STFUNC: /* delayed var ref */
    SST_IDP(stkptr, S_IDENT);
    (void)mkvarref(stkptr, SST_ENDG(stkptr));
    goto again;

  case S_DERIVED:
    if (SST_DBEGG(stkptr)) {
      (void)mkvarref(stkptr, SST_DBEGG(stkptr));
      return 1;
    }
    sptr = SST_SYMG(stkptr);
    mkident(stkptr);
    SST_SYMP(stkptr, sptr);
    goto add_sym_arg;

  case S_CONST:
    SST_CVLENP(stkptr, 0);
    if (SST_DTYPEG(stkptr) == DT_HOLL) {
      SST_DTYPEP(stkptr, DT_INT);
      SST_IDP(stkptr, S_EXPR);
    } else {
      if (SST_DTYPEG(stkptr) == DT_WORD)
        SST_DTYPEP(stkptr, DT_INT);
      mkexpr(stkptr);
    }
    *dtype = SST_DTYPEG(stkptr);
    return 1;

  case S_ACONST:
    /* resolve it */
    if (!SST_ACLG(stkptr)) { /* zero-sized array */
      int sdtype;
      sptr = sym_get_array("zs", "array", SST_DTYPEG(stkptr), 1);
      sdtype = DTYPEG(sptr);
      ADD_LWBD(sdtype, 0) = ADD_LWAST(sdtype, 0) = astb.bnd.one;
      ADD_UPBD(sdtype, 0) = ADD_UPAST(sdtype, 0) = astb.bnd.zero;
      ADD_EXTNTAST(sdtype, 0) =
          mk_extent(ADD_LWAST(sdtype, 0), ADD_UPAST(sdtype, 0), 0);
    } else {
      sptr = init_sptr_w_acl(0, SST_ACLG(stkptr));
    }
    mkident(stkptr);
    SST_SYMP(stkptr, sptr);
    goto add_const_sym_arg;

  case S_IDENT:
    /* resolve it */
    sptr = SST_SYMG(stkptr);
    switch (STYPEG(sptr)) {
    case ST_PD:
      sp2 = sptr;
      if (!EXPSTG(sptr)) {
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        sem_set_storage_class(sptr);
        goto add_sym_arg;
      }
      goto common_intrinsic;
    case ST_ENTRY:
      if (gbl.rutype == RU_SUBR && (flg.recursive || RECURG(sptr)))
        ;
      else if (gbl.rutype != RU_FUNC)
        error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      else if (!RESULTG(sptr))
        sptr = ref_entry(sptr);
      /* if RESULTG is set, the reference to the function name
       * must mean the function itself; if not, the function name
       * must mean the function result variable */
      goto add_sym_arg;
    case ST_UNKNOWN:
    case ST_IDENT:
      STYPEP(sptr, ST_VAR);
      FLANG_FALLTHROUGH;
    case ST_VAR:
    case ST_ARRAY:
      if (DTY(DTYPEG(sptr)) != TY_ARRAY || DDTG(DTYPEG(sptr)) == DT_DEFERCHAR ||
          DDTG(DTYPEG(sptr)) == DT_DEFERNCHAR) {
        /* test for scalar pointer */
        if ((ALLOCATTRG(sptr) || POINTERG(sptr)) && SDSCG(sptr) == 0 &&
            !F90POINTERG(sptr)) {
          if (SCG(sptr) == SC_NONE)
            SCP(sptr, SC_BASED);
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
        }
      }
      if (!SDSCG(sptr) && ASSUMRANKG(sptr)) {
        get_static_descriptor(sptr);
        ASSUMRANKP(SDSCG(sptr), 1);
      }
      goto add_sym_arg;
    case ST_USERGENERIC:
      if (GSAMEG(sptr)) {
        /* use the specific of the same name */
        sptr = GSAMEG(sptr);
        goto add_sym_arg;
      }
      /* can't pass the generic name as an argument */
      error(73, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      SST_DTYPEP(stkptr, *dtype = DT_INT);
      return 1;
    case ST_GENERIC:
      /* Generic used as an actual argument.  Use specific of same name.
       * If none, then assume its a variable unless generic is frozen.
       */
      sp2 = select_gsame(sptr); /* intrinsic of same name */
      if (sp2 == 0 || !EXPSTG(sptr)) {
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        sem_set_storage_class(sptr);
        goto add_sym_arg;
      }
      sp2 = intrinsic_as_arg(sptr);
      if (sp2 == 0) {
        /* may not be passed as argument */
        error(73, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SST_DTYPEP(stkptr, *dtype = DT_INT);
        return 1;
      }
      if (STYPEG(sp2) == ST_PROC)
        sptr = sp2;
      else if (sp2 != GSAMEG(sptr)) {
        DTYPEP(sp2, INTTYPG(sp2));
        sptr = sp2;
      } else
        DTYPEP(sptr, INTTYPG(sp2));
      goto add_sym_arg;
    case ST_INTRIN:
      sp2 = sptr;
      if (!EXPSTG(sptr)) {
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        sem_set_storage_class(sptr);
        goto add_sym_arg;
      }
    common_intrinsic:
      sp2 = intrinsic_as_arg(sptr);
      if (sp2 == 0) {
        /* may not be passed as argument */
        error(73, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SST_DTYPEP(stkptr, *dtype = DT_INT);
        return 1;
      }
      if (STYPEG(sp2) != ST_PROC)
        DTYPEP(sp2, INTTYPG(sp2));
      sptr = sp2;
      goto add_sym_arg;
    case ST_PROC:
      sp2 = SCOPEG(sptr);
      if (STYPEG(sp2) == ST_ALIAS)
        sp2 = SYMLKG(sp2);
      if (ELEMENTALG(sptr)) {
        error(464, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        SST_DTYPEP(stkptr, *dtype = DTYPEG(sptr));
        return 1;
      }
      TYPDP(sptr, 1); /* force it to appear in an EXTERNAL stmt */
      goto add_sym_arg;
    case ST_STFUNC:
    case ST_STRUCT:
    case ST_TYPEDEF:
      goto add_sym_arg;

    default:
      error(84, 3, gbl.lineno, SYMNAME(sptr), CNULL);
      SST_DTYPEP(stkptr, *dtype = DTYPEG(sptr));
      SST_ASTP(stkptr, mk_id(sptr));
      SST_SHAPEP(stkptr, A_SHAPEG(SST_ASTG(stkptr)));
      return 3;
    }

  case S_LVALUE:
    *dtype = SST_DTYPEG(stkptr);
    ARGP(SST_LSYMG(stkptr), 1);
    ast = SST_ASTG(stkptr);
    if (ast && A_TYPEG(ast) == A_MEM) {
      /* this is a derived-type member reference, see if the
       * member needs a static descriptor */
      sptr = find_pointer_variable(ast);
      if (DTY(DTYPEG(sptr)) != TY_ARRAY) {
        if (POINTERG(sptr) && SDSCG(sptr) == 0 && !F90POINTERG(sptr)) {
          if (STYPEG(sptr) != ST_MEMBER && SCG(sptr) == SC_NONE)
            SCP(sptr, SC_BASED);
          get_static_descriptor(sptr);
          get_all_descriptors(sptr);
        }
      }
    }
    sptr = SST_LSYMG(stkptr);
    SST_CVLENP(stkptr, 0);
    dt = DDTG(DTYPEG(sptr)); /* element dtype record */
    if ((DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR) && ADJLENG(sptr)) {
      SST_CVLENP(stkptr, size_ast(sptr, dt));
    }
    return 1;

  case S_EXPR:
  case S_LOGEXPR:
    *dtype = SST_DTYPEG(stkptr);
    if (flg.endian) {
      switch (DTY(SST_DTYPEG(stkptr))) {
      case TY_BINT:
      case TY_BLOG:
      case TY_SINT:
      case TY_SLOG:
        return tempify(stkptr);
      default:
        break;
      }
    }
    return 1;

  case S_REF:
    *dtype = SST_DTYPEG(stkptr);
    return 1;

  case S_VAL:
    *dtype = SST_DTYPEG(stkptr);
    return 1;

  case S_LABEL:
    *dtype = 0;
    return -SST_SYMG(stkptr);

  case S_SCONST:
    if (!SST_ACLG(stkptr)) {
      sptr = getcctmp_sc('d', sem.dtemps++, ST_VAR, SST_DTYPEG(stkptr), sem.sc);
    } else {
      sptr = init_derived_w_acl(0, SST_ACLG(stkptr));
    }
    mkident(stkptr);
    SST_SYMP(stkptr, sptr);
    goto add_const_sym_arg;

  case S_KEYWORD:
    return mkarg(SST_E3G(stkptr), dtype);

  case S_STAR:
  case S_TRIPLE:
    error(
        155, 3, gbl.lineno,
        "An argument to this subprogram looks like an array section subscript",
        CNULL);
    /* change to constant zero, see if we can avoid further errors */
    SST_DTYPEP(stkptr, *dtype = DT_INT);
    SST_ASTP(stkptr, astb.i0);
    SST_SHAPEP(stkptr, 0);
    SST_IDP(stkptr, S_CONST);
    return 3;

  default:
    interr("mkarg: arg has bad stkid", SST_IDG(stkptr), 3);
    return 3;
  }

add_sym_arg:
  sptr = ref_object(sptr);
add_const_sym_arg:
  ARGP(sptr, 1);
  SST_DTYPEP(stkptr, *dtype = DTYPEG(sptr));
  SST_ASTP(stkptr, mk_id(sptr));
  SST_SHAPEP(stkptr, A_SHAPEG(SST_ASTG(stkptr)));
  SST_CVLENP(stkptr, 0);
  dt = DDTG(DTYPEG(sptr)); /* element dtype record */
  if ((DTY(dt) == TY_CHAR || DTY(dt) == TY_NCHAR) && ADJLENG(sptr)) {
    SST_CVLENP(stkptr, size_ast(sptr, dt));
  }
  return 1;
}

#if defined(TARGET_WIN_X86) && defined(PGFTN)
/*
 * convert to upper case
 */
static void
upcase_name(char *name)
{
  char *p;
  int ch;
  for (p = name; ch = *p; ++p)
    if (ch >= 'a' && ch <= 'z')
      *p = ch + ('A' - 'a');
}
#endif

int
intrinsic_as_arg(int intr)
{
  int sp2;
  int cp;
  FtnRtlEnum rtlRtn;

  sp2 = intr;
  switch (STYPEG(intr)) {
  case ST_GENERIC:
    sp2 = select_gsame(intr);
    if (sp2 == 0)
      return 0;
    FLANG_FALLTHROUGH;
  case ST_PD:
  case ST_INTRIN:
    cp = PNMPTRG(sp2);
    if (cp == 0 || stb.n_base[cp] == '-')
      return 0;
    if (stb.n_base[cp] != '*' || stb.n_base[++cp] != '\0') {
      int dt;

      dt = INTTYPG(sp2);

      switch (INTASTG(sp2)) {
      case I_INDEX:
      case I_KINDEX:
      case I_NINDEX:
        if (XBIT(58, 0x40)) { /* input is f90 */
#ifdef CREFP
          if (WINNT_CREF) {
            rtlRtn = WINNT_NOMIXEDSTRLEN ? RTE_indexx_cr_nma : RTE_indexx_cra;
          } else
#endif
          {
            rtlRtn = RTE_indexxa;
          }
        } else if (XBIT(124, 0x10)) { /* -i8 for f77 */
          sp2 = intast_sym[I_KINDEX];
          dt = DT_INT8;
          rtlRtn = RTE_lenDsc;
        } else {
          rtlRtn = RTE_indexDsc;
        }
        break;
      case I_LEN:
      case I_ILEN:
      case I_KLEN:
        if (XBIT(58, 0x40)) { /* input is f90 */
#ifdef CREFP
          if (WINNT_CREF) {
            rtlRtn = WINNT_NOMIXEDSTRLEN ? RTE_lenx_cr_nma : RTE_lenx_cra;
          } else
#endif
          {
            rtlRtn = RTE_lenxa;
          }
        } else if (XBIT(124, 0x10)) { /* -i8 for f77 */
          sp2 = intast_sym[I_KLEN];
          dt = DT_INT8;
          rtlRtn = RTE_lenDsc;
          break;
        } else {
          rtlRtn = RTE_lenDsc;
        }
        break;
      }
      sp2 = sym_mkfunc(mkRteRtnNm(rtlRtn), dt);
      TYPDP(sp2, 1); /* force it to appear in an EXTERNAL stmt */
      if (WINNT_CALL)
        MSCALLP(sp2, 1);
#ifdef CREFP
      if (WINNT_CREF) {
        CREFP(sp2, 1);
        CCSYMP(sp2, 1);
      }
#endif
      break;
    }
    if (XBIT(124, 0x10)) { /* -i8 */
      switch (INTASTG(sp2)) {
      case I_IABS:
        sp2 = intast_sym[I_KIABS];
        break;
      case I_IDIM:
        sp2 = intast_sym[I_KIDIM];
        break;
      case I_MOD:
        sp2 = intast_sym[I_KMOD];
        break;
      case I_NINT:
        sp2 = intast_sym[I_KNINT];
        break;
      case I_IDNINT:
        sp2 = intast_sym[I_KIDNNT];
        break;
      case I_ISIGN:
        sp2 = intast_sym[I_KISIGN];
        break;
      /*
       * For the following, the integer specifics have been changed
       * to their corresponding integer*8 versions; however, the
       * function names are still refer to the integer forms.
       * Need to returning the sptr of the integer*8 intrinsic
       * so that the function name is correctly constructed.
       */
      case I_KIABS:
        sp2 = intast_sym[I_KIABS];
        break;
      case I_KIDIM:
        sp2 = intast_sym[I_KIDIM];
        break;
      case I_KIDNNT:
        sp2 = intast_sym[I_KIDNNT];
        break;
      case I_KISIGN:
        sp2 = intast_sym[I_KISIGN];
        break;
      default:
        break;
      }
    }
    if (XBIT(124, 0x8)) { /* -r8 */
      switch (INTASTG(sp2)) {
      case I_ALOG:
        sp2 = intast_sym[I_DLOG];
        break;
      case I_ALOG10:
        sp2 = intast_sym[I_DLOG10];
        break;
      case I_CABS:
        sp2 = intast_sym[I_CDABS];
        break;
      case I_AMOD:
        sp2 = intast_sym[I_DMOD];
        break;
      case I_ABS:
        sp2 = intast_sym[I_DABS];
        break;
      case I_SIGN:
        sp2 = intast_sym[I_DSIGN];
        break;
      case I_DIM:
        sp2 = intast_sym[I_DDIM];
        break;
      case I_SQRT:
        sp2 = intast_sym[I_DSQRT];
        break;
      case I_EXP:
        sp2 = intast_sym[I_DEXP];
        break;
      case I_SIN:
        sp2 = intast_sym[I_DSIN];
        break;
      case I_COS:
        sp2 = intast_sym[I_DCOS];
        break;
      case I_TAN:
        sp2 = intast_sym[I_DTAN];
        break;
      case I_AINT:
        sp2 = intast_sym[I_DINT];
        break;
      case I_ANINT:
        sp2 = intast_sym[I_DNINT];
        break;
      case I_ASIN:
        sp2 = intast_sym[I_DASIN];
        break;
      case I_ACOS:
        sp2 = intast_sym[I_DACOS];
        break;
      case I_ATAN:
        sp2 = intast_sym[I_DATAN];
        break;
      case I_SINH:
        sp2 = intast_sym[I_DSINH];
        break;
      case I_COSH:
        sp2 = intast_sym[I_DCOSH];
        break;
      case I_TANH:
        sp2 = intast_sym[I_DTANH];
        break;
      case I_ATAN2:
        sp2 = intast_sym[I_DATAN2];
        break;
      case I_SIND:
        sp2 = intast_sym[I_DSIND];
        break;
      case I_COSD:
        sp2 = intast_sym[I_DCOSD];
        break;
      case I_TAND:
        sp2 = intast_sym[I_DTAND];
        break;
      case I_AIMAG:
        sp2 = intast_sym[I_DIMAG];
        break;
      case I_ASIND:
        sp2 = intast_sym[I_DASIND];
        break;
      case I_ACOSD:
        sp2 = intast_sym[I_DACOSD];
        break;
      case I_ATAND:
        sp2 = intast_sym[I_DATAND];
        break;
      case I_ATAN2D:
        sp2 = intast_sym[I_DATAN2D];
        break;
      case I_CSQRT:
        sp2 = intast_sym[I_CDSQRT];
        break;
      case I_CLOG:
        sp2 = intast_sym[I_CDLOG];
        break;
      case I_CEXP:
        sp2 = intast_sym[I_CDEXP];
        break;
      case I_CSIN:
        sp2 = intast_sym[I_CDSIN];
        break;
      case I_CCOS:
        sp2 = intast_sym[I_CDCOS];
        break;
      case I_CONJG:
        sp2 = intast_sym[I_DCONJG];
        break;
      /*
       * For the following, the real/complex specifics have been changed
       * to their corresponding double real/complex versions; however, the
       * function names are still refer to the real/complex forms.
       * Need to returning the sptr of the 'double real/complex' intrinsic
       * so that the function name is correctly constructed.
       */
      case I_DLOG:
        sp2 = intast_sym[I_DLOG];
        break;
      case I_DLOG10:
        sp2 = intast_sym[I_DLOG10];
        break;
      case I_CDABS:
        sp2 = intast_sym[I_CDABS];
        break;
      case I_DMOD:
        sp2 = intast_sym[I_DMOD];
        break;
      case I_CDSQRT:
        sp2 = intast_sym[I_CDSQRT];
        break;
      case I_CDLOG:
        sp2 = intast_sym[I_CDLOG];
        break;
      case I_CDEXP:
        sp2 = intast_sym[I_CDEXP];
        break;
      case I_CDSIN:
        sp2 = intast_sym[I_CDSIN];
        break;
      case I_CDCOS:
        sp2 = intast_sym[I_CDCOS];
        break;
      }
    }
    break;
  default:;
  }
  if (SYMNAME(sp2)[0] != '.')
    TYPDP(sp2, 1); /* force it to appear in an INTRINSIC statement */
  return sp2;
}

/** \brief Performing checking on an argument to determine if it needs to be
 *         "protected".
 *  \param stkptr a stack entry representing a subprogram argument
 *  \param dtype  used to pass out data type of argument
 *  \return       sptr for alternate return label
 *
 *  When an argument needs to be protected, its value may be have to be stored
 *  in a temp and then the temp's address becomes the actual argument.  This
 *  occurs when a parenthesized expression is an actual argument.  NOTE that
 *  the logic in chkarg relies on an argument being `<expression>`; a flag
 *  (SST_PARENG/P) in the semantic stack is used and is only defined for
 *  `<expression>`; the flag is cleared when `<expression> ::= ...` occurs and
 *  is set when `<primary> ::= ( <expression> )` occurs.
 */
int
chkarg(SST *stkptr, int *dtype)
{
  int argtyp;
  int sptr, sp2;

  if (SST_PARENG(stkptr)) {
    switch (SST_IDG(stkptr)) {
    case S_CONST:
      argtyp = SST_DTYPEG(stkptr);
      if (argtyp == DT_HOLL || argtyp == DT_WORD || argtyp == DT_DWORD)
        argtyp = DT_INT;
      break;

    case S_ACONST:
      /* just let mkarg() deal with it */
      goto call_mkarg;

    case S_IDENT:
      /* resolve it */
      sptr = SST_SYMG(stkptr);
      switch (STYPEG(sptr)) {
      case ST_ENTRY:
        if (gbl.rutype != RU_FUNC)
          goto call_mkarg;
        sptr = ref_entry(sptr);
        goto store_var;
      case ST_UNKNOWN:
      case ST_IDENT:
        STYPEP(sptr, ST_VAR);
        FLANG_FALLTHROUGH;
      case ST_VAR:
      store_var:
        argtyp = DTYPEG(sptr);
        if (!DT_ISSCALAR(argtyp)) {
          /* could issue error message */
          goto call_mkarg;
        }
        if (argtyp == DT_ASSCHAR || argtyp == DT_ASSNCHAR) {
          /* could issue error message */
          goto call_mkarg;
        }
        break;
      case ST_USERGENERIC:
        if (GSAMEG(sptr)) {
          sptr = GSAMEG(sptr);
          goto call_mkarg;
        }
        /* make a scalar symbol here */
        sptr = newsym(sptr);
        STYPEP(sptr, ST_VAR);
        sem_set_storage_class(sptr);
        goto store_var;
      case ST_GENERIC:
        /* Generic used as an actual argument.  Use specific of same
         * name. If none, then assume its a variable unless generic is
         * frozen.
         */
        sp2 = select_gsame(sptr); /* intrinsic of same name */
        if (sp2 == 0 || !EXPSTG(sptr)) {
          sptr = newsym(sptr);
          STYPEP(sptr, ST_VAR);
          sem_set_storage_class(sptr);
          goto store_var;
        }
        goto common_intrinsic;
      /* fall through to ... */
      case ST_INTRIN:
      case ST_PD:
        sp2 = sptr;
        if (!EXPSTG(sptr)) {
          sptr = newsym(sptr);
          STYPEP(sptr, ST_VAR);
          sem_set_storage_class(sptr);
          goto store_var;
        }
      common_intrinsic:
      /* fall thorugh to ... */
      case ST_PROC:
      case ST_STFUNC:
      case ST_STRUCT:
      case ST_ARRAY:
        goto call_mkarg;

      default:
        goto call_mkarg;
      }
      break;

    case S_LVALUE:
      argtyp = SST_DTYPEG(stkptr);
      if (!DT_ISSCALAR(argtyp)) {
        /* perhaps, we could issue an error message */
        goto call_mkarg;
      }
      if (DTY(argtyp) == TY_CHAR || DTY(argtyp) == TY_NCHAR) {
        /* we can't do much about char lvalues; if a substring, we
         * don't pass up the new length.  This will lead to an incorrect
         * length for the temporary.  Also, the same is true if it's a
         * subscripted passed length char array.
         * EVENTUALLY, will probably allocate a temp, and use the
         * temp's address.
         */
        goto call_mkarg;
      }
      break;
    default:
      goto call_mkarg;
    }
    /*
     *  must parenthesize the argument.
     */
    *dtype = argtyp;
    (void)mkexpr(stkptr);
    /* always generate parens */
    SST_ASTP(stkptr, mk_paren((int)SST_ASTG(stkptr), argtyp));
    return 1;
  }
  if (SST_IDG(stkptr) == S_EXPR) {
    /* need to protect a scalar expression whose ast is a reference;
     * an example is if the expression was 'id + 0', which is reduced
     * to just 'id'
     */
    int ast;
    ast = SST_ASTG(stkptr);
    if (DT_ISSCALAR(A_DTYPEG(ast)) && A_ISLVAL(A_TYPEG(ast)) &&
        (A_TYPEG(ast) != A_ID || !POINTERG(A_SPTRG(ast)))) {
      ast = mk_paren(ast, (int)A_DTYPEG(ast));
      SST_ASTP(stkptr, ast);
      ;
      return 1;
    }
  }

call_mkarg:
  return (mkarg(stkptr, dtype));
}

/** \brief Allocate a temporary, assign it the value, and return the temp's
 *         base.
 */
int
tempify(SST *stkptr)
{
  int argtyp;
  SST tmpsst;
  int tmpsym;

  argtyp = SST_DTYPEG(stkptr);
  tmpsym = get_temp(argtyp);
  mkident(&tmpsst);
  SST_SYMP(&tmpsst, tmpsym);
  SST_LSYMP(&tmpsst, tmpsym);
  SST_DTYPEP(&tmpsst, argtyp);
  SST_SHAPEP(&tmpsst, 0);
  (void)add_stmt(assign(&tmpsst, stkptr));
  mkexpr(&tmpsst);
  *stkptr = tmpsst;
  return 1;
}

/*---------------------------------------------------------------------*/

/* A function entry is referenced where the intent is to reference the
 * "local" variable (a ccsym created for the result).  Since entries may
 * have different types, a "local" variable which is compatible for the
 * data type is found; if not, one is created.
 */
int
ref_entry(int ent)
{
  int fval;
  int dtype;

  fval = FVALG(ent);
  if (fval) {
    if (DCLDG(ent) && !DCLDG(fval))
      DTYPEP(fval, DTYPEG(ent)); /* watch out for type after function/entry */
  } else {
    dtype = DTYPEG(ent);
    if (DTY(dtype) == TY_ARRAY) {
      fval = get_fval_array(ent);
    } else {
      fval = insert_sym(ent);
      pop_sym(fval);
      if (POINTERG(ent)) {
        HCCSYMP(fval, 1);
      }
      SCP(fval, SC_DUMMY); /* so optimizer doesn't delete */
      DCLDP(fval, 1);      /* so 'undeclared' messages don't occur */
      if (dtype == DT_NONE) {
        /* an error message has been issued; just set dtype */
        dtype = DT_INT;
      }
      DTYPEP(fval, dtype);
      {
        STYPEP(fval, ST_VAR);
      }
      if (POINTERG(ent)) {
        POINTERP(fval, 1);
        F90POINTERP(fval, F90POINTERG(ent));
        if (!F90POINTERG(fval)) {
          get_static_descriptor(fval);
          get_all_descriptors(fval);
        }
        INTENTP(fval, INTENT_OUT);
      }
      ADJLENP(fval, ADJLENG(ent));
      ASSUMLENP(fval, ASSUMLENG(ent));
    }
    FVALP(ent, fval);
    if (STYPEG(ent) != ST_ENTRY) {
      /* prevent astout from processing */
      IGNOREP(fval, 1);
    }
  }
  if (sem.parallel || sem.task || sem.target || sem.teams
      || sem.orph
  ) {
    /* if in a parallel region, need to first determine if a private copy
     * was declared for the entry's variable in the parallel directive.
     * Then need to check the current scope for a default clause.
     */
    char *name;
    int new;
    if (ent == gbl.currsub) {
      name = SYMNAME(fval);
      new = getsymbol(name);
    } else {
      new = fval;
    }
    new = sem_check_scope(new, ent);
    if (new != ent)
      fval = new;
  }
  return fval;
}

/*---------------------------------------------------------------------*/

/** \brief Given a generic intrinsic, select the corresponding specific of the
 *         same name.
 *
 *  Normally, the selection is performed by just accessing the GSAME field of
 *  the generic.  When an option to override the meaning of INTEGER (-noi4) is
 *  selected, some analysis must be done to select the specific intrinsic
 *  whose argument type matches the type implied by the option.
 */
int
select_gsame(int gnr)
{
  int spec;

  if ((spec = GSAMEG(gnr)) == 0)
    return 0;
  if (ARGTYPG(spec) == DT_INT) {
    if (!flg.i4)
      spec = GSINTG(gnr);
    else if (XBIT(124, 0x10))
      spec = GINT8G(gnr);
  } else if (XBIT(124, 0x8)) {
    if (ARGTYPG(spec) == DT_REAL)
      spec = GDBLEG(gnr);
    else if (ARGTYPG(spec) == DT_CMPLX)
      spec = GDCMPLXG(gnr);
  }
  return spec;
}

/*---------------------------------------------------------------------*/

static int
get_fval_array(int ent)
{
  int sptr;
  int dtype;
  ADSC *ad;

  if (FVALG(ent)) {
    sptr = insert_sym(FVALG(ent));
  } else {
    sptr = insert_sym(ent);
  }
  pop_sym(sptr);
  dtype = DTYPEG(ent);
  HCCSYMP(sptr, 1);
  DCLDP(sptr, 1);
  SCOPEP(sptr, stb.curr_scope);
  DTYPEP(sptr, dtype);
  ADJLENP(sptr, ADJLENG(ent));
  ASSUMLENP(sptr, ASSUMLENG(ent));

  SCP(sptr, SC_DUMMY);
  INTENTP(sptr, INTENT_OUT);
  if (!POINTERG(ent)) {
    ad = AD_DPTR(dtype);
    if (AD_ADJARR(ad)) {
      ADJARRP(sptr, 1);
      ADJARRP(ent, 1);
    } else if (AD_DEFER(ad)) {
      ASSUMSHPP(sptr, 1);
      if (!XBIT(54, 2) && !(XBIT(58, 0x400000) && TARGETG(sptr)))
        SDSCS1P(sptr, 1);
      ASSUMSHPP(ent, 1);
    } else if (AD_ASSUMSZ(ad)) {
      ASUMSZP(sptr, 1);
      ASUMSZP(ent, 1);
      SEQP(sptr, 1);
    }
  }

  else {
    STYPEP(sptr, ST_ARRAY);
    if (POINTERG(ent)) {
      POINTERP(sptr, 1);
      F90POINTERP(sptr, F90POINTERG(ent));
      if (!F90POINTERG(sptr)) {
        get_static_descriptor(sptr);
        get_all_descriptors(sptr);
      }
    }
  }

  return sptr;
}

/*---------------------------------------------------------------------*/

/** \brief Make a keyword string for a ST_PROC. */
char *
make_kwd_str(int ext)
{
  char *kwd_str;
  kwd_str = make_keyword_str(PARAMCTG(ext), DPDSCG(ext));
  return kwd_str;
}

/** \brief Make a keyword string from the DPDSC auxiliary structure. */
char *
make_keyword_str(int paramct, int dpdsc)
{
  int cnt;
  int arg; /* argument sptr */
  int i;
  const char *name;
  int optional;
  int len;
  int size;
  int avl;
  char *kwd_str;
  int td;

  avl = 0;
  size = 100;
  NEW(kwd_str, char, size);
  for (cnt = paramct; cnt > 0; dpdsc++, cnt--) {
    if ((arg = *(aux.dpdsc_base + dpdsc))) {
      optional = OPTARGG(arg);
      name = SYMNAME(arg);
      len = strlen(name);
      td = CLASSG(arg) && CCSYMG(arg);
      if (HCCSYMG(arg) && DESCARRAYG(arg) && STYPEG(arg) == ST_DESCRIPTOR)
        td = 1;
    } else {
      /* alternate returns */
      optional = 0;
      name = "&";
      len = 1;
      td = 0;
    }
    i = avl;
    avl += (optional + len + 2); /* len chars in name, 1 for ' ', 1 for null */
    NEED(avl, kwd_str, char, size, size + 100);
    if (optional)
      kwd_str[i++] = '*';
    else if (td)
      kwd_str[i++] = '!';
    strcpy(kwd_str + i, name);
    if (cnt > 1)
      kwd_str[i + len] = ' ';
    avl--;
  }

  return kwd_str;
}

/*---------------------------------------------------------------------*/

/** \defgroup args  Positional and keyword arguments
 *
 *  Support for extracting positional or keyword arguments for an intrinsic.
 *  For each intrinsic, the symbol table utility has created a string which
 *  defines the arguments and their positions in the argument list and
 *  keywords.
 *
 *  Optional arguments are indicating by prefixing their keywords with '*'.
 *
 * @{
 */

static LOGICAL user_subr = FALSE; /**< TRUE if invoking a user subprogram */
static int nz_digit_str(char *);

/** \brief Extract the arguments from the semantic list into `sem.argpos[]` in
 *         positional order.
 *  \param list    list of arguments
 *  \param cnt     maximum number of arguments allowed for intrinsic
 *  \param kwdarg  string defining position and keywords of arguments
 */
LOGICAL
get_kwd_args(ITEM *list, int cnt, const char *kwdarg)
{
  return get_keyword_args(list, cnt, kwdarg, 0, 0);
}

/** \brief Similar to get_kwd_args but can also specify a pass-object dummy
 *         argument.
 *  \param list     list of arguments
 *  \param cnt      maximum number of arguments allowed for intrinsic
 *  \param kwdarg   string defining position and keywords of arguments
 *  \param pod      if set indicates there is a passed-object dummy argument
 *  \param pass_pos index of the passed-object dummy argument when pod is set
 */
static LOGICAL
get_keyword_args(ITEM *list, int cnt, const char *kwdarg, int pod, int pass_pos)
{
  SST *stkp;
  int pos;
  int i;
  const char *kwd, *np;
  int kwd_len;
  char *actual_kwd; /* name of keyword used with the actual arg */
  int actual_kwd_len;
  LOGICAL kwd_present;
  int varpos;
  int varbase;
  int pass_pos2 = 0, pod2 = 0; /* pass object info for type bound procedure */

  /* convention for the keyword 'variable' arguments ---
   * the keyword specifier is of the form
   *     #<pos>#<base>#<kwd>
   * where,
   *      <pos>  = digit indicating the zero-relative positional index where
   *               the variable arguments begin in the argument list.
   *      <base> = digit indicating value to be subtracted from the digit
   *               string suffix of the keyword.
   *      <kwd>  = name of the keyword which varies (i.e., the prefix).
   */

  kwd_present = FALSE;
  /* extra arguments may be stored in argpos[]; allow for 'cnt' extra args */
  sem.argpos = (argpos_t *)getitem(0, sizeof(argpos_t) * cnt * 2);

  for (i = 0; i < cnt * 2; i++) {
    ARG_STK(i) = NULL;
    ARG_AST(i) = 0;
  }

  if (!pod && !pass_pos) {
    pod2 = pass_object_dummy;
    pass_pos2 = pass_position;
  }
  pass_object_dummy = pass_position = 0;

  for (pos = 0; list != ITEM_END; pos++) {
    if (pod && pos == pass_pos) {
      /* examining the position of the passed-object dummy argument;
       * go to the next position, but do not move the list.
       */
      continue;
    }
    stkp = list->t.stkp;
    if (pod2 && pos == pass_pos2) {
      /* pass object for type bound procedure, so set the argument
       * and continue. Otherwise, we may get an error since the
       * pass argument does not have a keyword.
       */
      ARG_STK(pos) = stkp;
      ARG_AST(pos) = SST_ASTG(stkp);
      list = list->next;
      continue;
    }
    if (SST_IDG(stkp) == S_KEYWORD) {
      kwd_present = TRUE;
      actual_kwd = scn.id.name + SST_CVALG(stkp);
      actual_kwd_len = strlen(actual_kwd);
      kwd = kwdarg;
      for (i = 0; TRUE; i++) {
        varbase = 0; /* variable part not seen */
        if (*kwd == '*')
          kwd++;
        else if (*kwd == '#') {
          /*  #<pos>#<base>#<kwd>  */
          kwd++;
          varpos = *kwd - '0'; /* numerical value of <pos> */
          kwd += 2;
          varbase = *kwd; /* digit (char) to be subtracted */
          kwd += 2;
        } else if (strncmp(kwd, "_V_", 3) == 0 && kwd[3] != ' ' &&
                   kwd[3] != '\0') {
          /* Use the original argument name for VALUE dummy arguments
           * that have been renamed in semant.c to distinguish them from
           * their local copies.
           */
          kwd += 3;
        }
        kwd_len = 0;
        for (np = kwd; TRUE; np++, kwd_len++)
          if (*np == ' ' || *np == '\0')
            break;
        if (varbase && (i = nz_digit_str(actual_kwd + kwd_len)) &&
            strncmp(kwd, actual_kwd, kwd_len) == 0) {
          /* compute actual position as:
           *     <digit suffix> - <base> + <pos>
           */
          i = i - (varbase - '0') + varpos;
          if (i >= cnt)
            goto ill_keyword;
          break;
        }
        if (kwd_len == actual_kwd_len &&
            strncmp(kwd, actual_kwd, actual_kwd_len) == 0)
          break;
        if (*np == '\0')
          goto ill_keyword;
        kwd = np + 1; /* skip over blank */
      }
      if (ARG_STK(i))
        goto ill_keyword;
      stkp = SST_E3G(stkp);
      ARG_STK(i) = stkp;
      ARG_AST(i) = SST_ASTG(stkp);
    } else {
      if (kwd_present) {
        error(155, 3, gbl.lineno,
              "Positional arguments must not follow keyword arguments", CNULL);
        return TRUE;
      }
      if (ARG_STK(pos)) {
        char print[22];
        kwd = kwdarg;
        for (i = 0; TRUE; i++) {
          if (*kwd == '*' || *kwd == ' ')
            kwd++;
          if (*kwd == '#') {
            error(79, 3, gbl.lineno, kwd + 5, "...");
            return TRUE;
          }
          if (*kwd == '\0') {
            interr("get_keyword_args, kwdnfd", pos, 3);
            return TRUE;
          }
          kwd_len = 0;
          for (np = kwd; TRUE; np++) {
            if (*np == ' ' || *np == '\0')
              break;
            kwd_len++;
          }
          if (i == pos)
            break;
          kwd = np;
        }
        if (kwd_len > 21)
          kwd_len = 21;
        strncpy(print, kwd, kwd_len);
        print[kwd_len] = '\0';
        error(79, 3, gbl.lineno, print, CNULL);
        return TRUE;
      }
      ARG_STK(pos) = stkp;
      ARG_AST(pos) = SST_ASTG(stkp);
    }
    list = list->next;
  }

  /* determine if required argument is not present */

  kwd = kwdarg;
  for (pos = 0; pos < cnt; pos++, kwd = np) {
    if (*kwd == ' ')
      kwd++;
    if (*kwd == '#')
      break;
    kwd_len = 0;
    for (np = kwd; TRUE; np++) {
      if (*np == ' ' || *np == '\0')
        break;
      kwd_len++;
    }
    if (*kwd == '!') {
      if (ARG_STK(pos) == NULL)
        break; /* continue; */
      else
        error(155, 3, gbl.lineno, "Too many arguments specified for call",
              CNULL);
    }
    if (*kwd == '*')
      continue;
    if ((pod && pos == pass_pos) || (pod2 && pos == pass_pos2))
      /* don't check the position of the passed-object dummy argument */
      ;
    else if (ARG_STK(pos) == NULL) {
      char print[22];
      if (kwd_len > 21)
        kwd_len = 21;
      strncpy(print, kwd, kwd_len);
      print[kwd_len] = '\0';
      error(186, kwd_present || user_subr ? 3 : 1, gbl.lineno, print, CNULL);
      return TRUE;
    }
  }

  return FALSE;

ill_keyword:
  error(79, 3, gbl.lineno, actual_kwd, CNULL);
  return TRUE;
}

static int
nz_digit_str(char *s)
{
  int val;

  val = 0; /* not a nonzero digit string */
  for (; *s != '\0'; ++s)
    switch (*s) {
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      val = val * 10 + (*s - '0');
      break;
    default:
      return 0;
    }

  return val;
}

/** \brief  Call get_kwd_args and evaluate each argument.
 *  \param list    list of arguments
 *  \param cnt     maximum number of arguments allowed for intrinsic
 *  \param kwdarg  string defining position and keywords of arguments
 */
LOGICAL
evl_kwd_args(ITEM *list, int cnt, const char *kwdarg)
{
  SST *stkp;
  int i, sptr;

  if (get_kwd_args(list, cnt, kwdarg))
    return TRUE;

  for (i = 0; i < cnt; i++) {
    if ((stkp = ARG_STK(i))) {
      if (SST_IDG(stkp) == S_IDENT && (sptr = SST_SYMG(stkp)) &&
          STYPEG(sptr) == ST_PROC) {
        /* passing a procedure as an argument */
        SST_DTYPEP(stkp, DTYPEG(sptr));
        SST_LSYMP(stkp, sptr);
        SST_ASTP(stkp, mk_id(sptr));
      } else if (SST_IDG(stkp) == S_SCONST) {
        (void)mkarg(stkp, &SST_DTYPEG(stkp));
      } else {
        (void)mkexpr(stkp);
      }
      ARG_AST(i) = SST_ASTG(stkp);
    }
  }

  return FALSE;
}

/**
 * \param list  list of arguments
 * \param cnt   maximum number of arguments allowed for intrinsic
 *
 * Arguments are: array, base, indx1, ..., indxn, mask<br>
 * where n = rank of base, mask is optional
 *
 * Don't use get_kwd_args() since it is not designed to handle an optional
 * argument after a variable number of arguments.  Note that cnt is the
 * actual number of arguments; get_kwd_args() & evl_kwd_args() are passed the
 * the maximum number of arguments.
 *
 * If the arguments are correct, the output (ARG_STK) will be in the order:
 *
 *     0      - array
 *     1      - base
 *     2      - mask
 *     3      - indx1
 *     ...
 *     3+n-1  -  indxn
 */
LOGICAL
sum_scatter_args(ITEM *list, int cnt)
{
  SST *stkp;
  int pos;
  int i;
  char *actual_kwd; /* name of keyword used with the actual arg */
  int actual_kwd_len;
  LOGICAL kwd_present;
  SST *mask;
  int rank;

  kwd_present = FALSE;
  sem.argpos = (argpos_t *)getitem(0, sizeof(argpos_t) * cnt);

  for (i = 0; i < cnt; i++) {
    ARG_STK(i) = NULL;
    ARG_AST(i) = 0;
  }
  /*
   * first, place the arguments in the positional order per the spec
   * except that 'mask=arg' is a special case.  The positional order is
   *     array, base, indx1, ..., indxn, mask
   */
  mask = NULL; /* set only if keyword form of mask is seen */
  for (pos = 0; list != ITEM_END; list = list->next, pos++) {
    stkp = list->t.stkp;
    if (SST_IDG(stkp) == S_KEYWORD) {
      kwd_present = TRUE;
      actual_kwd = scn.id.name + SST_CVALG(stkp);
      actual_kwd_len = strlen(actual_kwd);
      if (strcmp("array", actual_kwd) == 0)
        i = 0;
      else if (strcmp("base", actual_kwd) == 0)
        i = 1;
      else if (strcmp("mask", actual_kwd) == 0) {
        mask = SST_E3G(stkp);
        continue;
      } else if (strncmp("indx", actual_kwd, 4) == 0) {
        if ((i = nz_digit_str(actual_kwd + 4))) {
          /* compute actual position as:
           *     <digit suffix> - <base> + <pos>
           */
          i = i + 1; /* positions 2, ..., 2+n-1 */
          if (i >= cnt)
            goto ill_keyword;
        }
      }
      if (ARG_STK(i))
        goto ill_keyword;
      stkp = SST_E3G(stkp);
      ARG_STK(i) = stkp;
      ARG_AST(i) = SST_ASTG(stkp);
    } else {
      if (ARG_STK(pos)) {
        const char *str;
        if (pos == 0)
          str = "array";
        else if (pos == 1)
          str = "base";
        else
          str = "indx or mask";
        error(79, 3, gbl.lineno, str, CNULL);
        return TRUE;
      }
      ARG_STK(pos) = stkp;
      ARG_AST(pos) = SST_ASTG(stkp);
    }
  }

  /* determine if required argument is not present */

  if (ARG_STK(0) == NULL) {
    error(186, kwd_present ? 3 : 1, gbl.lineno, "array", CNULL);
    return TRUE;
  }
  if (ARG_STK(1) == NULL) {
    error(186, kwd_present ? 3 : 1, gbl.lineno, "base", CNULL);
    return TRUE;
  }

  stkp = ARG_STK(1);
  mkexpr(stkp);
  rank = rank_of_ast(SST_ASTG(stkp));

  if (mask) {
    if (ARG_STK(cnt - 1)) {
      error(79, 3, gbl.lineno, "mask", CNULL);
      return TRUE;
    }
    if (rank != cnt - 3) {
      error(186, kwd_present ? 3 : 1, gbl.lineno, "indx...", CNULL);
      return TRUE;
    }
  } else if (rank < cnt - 2) {
    mask = ARG_STK(cnt - 1);
    if (rank != cnt - 3) {
      error(186, kwd_present ? 3 : 1, gbl.lineno, "indx...", CNULL);
      return TRUE;
    }
  } else if (rank != cnt - 2) {
    error(186, kwd_present ? 3 : 1, gbl.lineno, "indx...", CNULL);
    return TRUE;
  }

  /* reposition the indx arguments so that they appear at the end */

  for (i = 2 + rank; i > 2; i--) {
    ARG_STK(i) = ARG_STK(i - 1);
    ARG_AST(i) = ARG_AST(i - 1);
  }

  ARG_STK(2) = mask;
  if (mask)
    ARG_AST(2) = SST_ASTG(mask);
  else
    ARG_AST(2) = 0;

  /* now, evaluate the arguments */

  for (i = 2 + rank; i >= 0; i--) {
    if ((stkp = ARG_STK(i))) {
      (void)mkexpr(stkp);
      ARG_AST(i) = SST_ASTG(stkp);
    }
  }

  return FALSE;

ill_keyword:
  error(79, 3, gbl.lineno, actual_kwd, CNULL);
  return TRUE;
}

/**@}*/

/*---------------------------------------------------------------------*/

/** \brief Process information for deferred interface argument checking in
 *         in the compat_arg_lists() function below.
 *
 *   If the performChk argument is false, then we save the information
 *   (defer the check). If performChk argument is true, then we perform
 *   the argument checking. Note: If performChk is true, then the other
 *   arguments are ignored.
 *
 * \param formal is the symbol table pointer of the dummy/formal argument.
 * \param actual is the symbol table pointer of the actual argument.
 * \param flags are comparison flags that enable/disable certain checks
 * \param lineno is the source line number for the deferred check
 * \param performChk is false to defer checks and true to perform the checks.
 */
void
defer_arg_chk(SPTR formal, SPTR actual, SPTR subprog, 
              cmp_interface_flags flags, int lineno, bool performChk)
{

  typedef struct chkList {
    char *formal;
    SPTR actual;
    char *subprog;
    cmp_interface_flags flags;
    int lineno;
    struct chkList * next;
  }CHKLIST;

  static CHKLIST *list = NULL;
  CHKLIST *ptr, *prev;

  if (!performChk) {
    /* Add a deferred check to the list */
    NEW(ptr, CHKLIST, sizeof(CHKLIST));
    NEW(ptr->formal, char, strlen(SYMNAME(formal))+1);
    strcpy(ptr->formal, SYMNAME(formal));
    ptr->actual = actual;
    NEW(ptr->subprog, char, strlen(SYMNAME(subprog))+1);
    strcpy(ptr->subprog, SYMNAME(subprog));
    ptr->flags = flags;
    ptr->lineno = lineno;
    ptr->next = list;
    list = ptr;
  } else if (sem.which_pass == 1) {
    for(prev = ptr = list; ptr != NULL; ) {
      if (strcmp(SYMNAME(gbl.currsub),ptr->subprog) == 0) { 
          /* perform argument check */
          formal = getsym(ptr->formal, strlen(ptr->formal));
          if (!compatible_characteristics(formal, ptr->actual, ptr->flags)) {
            char details[1000];
            sprintf(details, "- arguments of %s and %s do not agree",
                    SYMNAME(ptr->actual), ptr->formal);
            error(74, 3, ptr->lineno, ptr->subprog, details);
          }
          if (prev == ptr) {
            prev = ptr->next;
            FREE(ptr->formal);
            FREE(ptr->subprog);
            FREE(ptr);
            list = ptr = prev;
          } else {
            prev->next = ptr->next;
            FREE(ptr->formal);
            FREE(ptr->subprog);
            FREE(ptr);
            ptr = prev->next;
          }
       } else {
         prev = ptr;
         ptr = ptr->next;
      }
    }
  }

}
    
      
   
/** \brief For arguments that are subprograms, check that their argument lists
 *         are compatible.
 */
static LOGICAL
compat_arg_lists(int formal, int actual)
{
  int paramct;
  int fdscptr, adscptr;
  int i;
  bool func_chk;
  cmp_interface_flags flags;

  /* TODO: Not checking certain cases for now. */
  if (STYPEG(actual) == ST_INTRIN || STYPEG(actual) == ST_GENERIC)
    return TRUE;

  flags = (IGNORE_ARG_NAMES | RELAX_STYPE_CHK | RELAX_POINTER_CHK | 
           RELAX_PURE_CHK_2);
  func_chk = (STYPEG(formal) == ST_PROC && STYPEG(actual) == ST_PROC && 
             FVALG(formal) &&  FVALG(actual));

  if (func_chk && resolve_sym_aliases(SCOPEG(SCOPEG(formal))) == gbl.currsub){
       flags |= DEFER_IFACE_CHK;
  } 

  if (func_chk && !compatible_characteristics(formal, actual, flags)) {
    return FALSE;
  }

  if (flags & DEFER_IFACE_CHK) {
    /* We are calling an internal subprogram. We need to defer the 
     * check on the procedure dummy argument until we have seen the 
     * internal subprogram.
     */
    defer_arg_chk(formal, actual, SCOPEG(formal), (flags ^ DEFER_IFACE_CHK), 
                  gbl.lineno, false); 
  }

  fdscptr = DPDSCG(formal);
  adscptr = DPDSCG(actual);
  if (fdscptr == 0 || adscptr == 0 || (flags & DEFER_IFACE_CHK)) {
    return TRUE; /* No dummy parameter descriptor; can't check. */
  }
  paramct = PARAMCTG(formal);
  if (PARAMCTG(actual) != paramct)
    return FALSE;
  for (i = 0; i < paramct; i++, fdscptr++, adscptr++) {
    int farg, aarg;

    farg = *(aux.dpdsc_base + fdscptr);
    aarg = *(aux.dpdsc_base + adscptr);
    if (STYPEG(farg) == ST_PROC) {
      if (STYPEG(aarg) != ST_PROC && STYPEG(aarg) != ST_ENTRY &&
          STYPEG(aarg) != ST_INTRIN && STYPEG(aarg) != ST_GENERIC)
        return FALSE;
      if (!compat_arg_lists(farg, aarg))
        return FALSE;
      /* If not functions, don't try to check return type. */
      if (!DCLDG(farg) && !FUNCG(farg) && !DCLDG(aarg) && !FUNCG(aarg))
        continue;
    }
    if (!cmpat_dtype_with_size(DTYPEG(farg), DTYPEG(aarg)))
      return FALSE;
  }
  return TRUE;
}

/** \brief Check arguments passed to a user subprogram which has an interface
 *         block. Its keyword string is available and is located by kwd_str.
 */
LOGICAL
check_arguments(int ext, int count, ITEM *list, char *kwd_str)
{
  int dpdsc;
  int paramct;
  paramct = PARAMCTG(ext);
  dpdsc = DPDSCG(ext);
  return chk_arguments(ext, count, list, kwd_str, paramct, dpdsc, 0, NULL);
}
/** \brief Check arguments passed to a user subprogram which has an interface
 *         block. Its keyword string is available and is located by kwd_str.
 *
 *  NOTE: if callee is non-zero, the call is via some procedure pointer, and
 *  callee is the ast of pointer and ext will be the sptr of the pointer
 *  variable or member. Otherwise, callee is 0 and ext is the sptr of the
 *  subroutine/function.
 */
LOGICAL
chk_arguments(int ext, int count, ITEM *list, char *kwd_str, int paramct,
              int dpdsc, int callee, int *p_pass_pos)
{
  int i;

  if (p_pass_pos)
    *p_pass_pos = -1; /* < 0 = > NO passed object */
  if (count > paramct) {
    error(187, 3, gbl.lineno, SYMNAME(ext), CNULL);
    return TRUE;
  }
  if (callee == 0 || A_TYPEG(callee) != A_MEM || !PASSG(ext)) {
    user_subr = TRUE;
    if (get_kwd_args(list, paramct, kwd_str)) {
      user_subr = FALSE;
      return TRUE;
    }
  } else {
    /* component procedure pointer with a passed-object dummy argument */
    int pass_pos = 0;
    int pdum;

    pdum = PASSG(ext);
    if (!tk_match_arg(DTYPEG(pdum), A_DTYPEG(A_PARENTG(callee)), TRUE)) {
      error(155, 3, gbl.lineno,
            "Type mismatch for the passed-object dummy argument",
            SYMNAME(pdum));
      return TRUE;
    }
    for (i = 0; i < paramct; i++) {
      if (pdum == aux.dpdsc_base[dpdsc + i]) {
        pass_pos = i;
        break;
      }
    }
    if (pass_pos >= paramct) {
      /* This should not happen since semant has already searched for
       * the passed-object dummy.  Just call it a type mismatch error!.
       */
      error(155, 3, gbl.lineno,
            "Type mismatch for the passed-object dummy argument",
            SYMNAME(pdum));
      return TRUE;
    }
    user_subr = TRUE;
    if (get_keyword_args(list, paramct, kwd_str, 1, pass_pos)) {
      user_subr = FALSE;
      return TRUE;
    }
    *p_pass_pos = pass_pos;
  }
  user_subr = FALSE;

  for (i = 0; i < paramct; i++, dpdsc++) {
    SST *sp;
    int dum;
    int actual;
    int arg;
    char buf[32];
    int sptr;

    sprintf(buf, "%d", i + 1); /* prepare for error messages */
    if ((sp = ARG_STK(i))) {
      (void)chkarg(sp, &dum);
      XFR_ARGAST(i);
    }
    actual = ARG_AST(i);
    arg = aux.dpdsc_base[dpdsc];

    if (arg) {
      if (!actual) {
        /* optional argument not present; store in the ast entry
         * the number of ast arguments which must be filled in with
         * the 'null' pointer (astb.ptr0).
         */
        ARG_AST(i) = 1;
      } else {
        int ddum, dact, elddum, eldact;
        int shape;
        LOGICAL dum_is_proc;

        if (STYPEG(arg) == ST_ENTRY || STYPEG(arg) == ST_PROC) {
          dum_is_proc = TRUE;
          if (FVALG(arg))
            ddum = DTYPEG(FVALG(arg));
          else
            ddum = DTYPEG(arg);
        } else {
          dum_is_proc = FALSE;
          ddum = DTYPEG(arg);
        }
        elddum = DDTG(ddum);
        dact = A_DTYPEG(actual);
        eldact = DDTG(dact);
        shape = A_SHAPEG(actual);
        if (DTY(eldact) == TY_PTR && DTY(elddum) == TY_PROC) {
          eldact = DTY(eldact + 1);
          eldact = DDTG(eldact);
        } else if (DTY(eldact) == TY_PROC && DTY(elddum) == TY_PTR) {
          elddum = DTY(elddum + 1);
          elddum = DDTG(elddum);
        } else if (dum_is_proc && DTY(eldact) == TY_PTR) {
          eldact = DTY(eldact + 1);
          if (DTY(eldact) == TY_PROC && (DTY(eldact + 5) || DTY(eldact + 2))) {
            /* Get eldact from either the result variable (i.e., DTY(eldact+5))
             * or interface (i.e., DTY(eldact+2)).
             */ 
            int ss;
            ss = DTY(eldact + 5) ? DTY(eldact + 5) : DTY(eldact + 2);
            if (FVALG(ss))
              eldact = DTYPEG(FVALG(ss));
            else
              eldact = DTYPEG(ss);
          }
          eldact = DDTG(eldact);
        }
        if (STYPEG(arg) == ST_ARRAY) {
          if (shape == 0) {
            int tmp;
            tmp = actual;
            if (A_TYPEG(tmp) == A_SUBSTR)
              tmp = A_LOPG(tmp);
            if (ASSUMSHPG(arg) && !ASSUMRANKG(arg)) {
              /* if the dummy is assumed-shape,
               * the user is trying to pass a scalar, constant
               * or array element into an assumed-shape array
               *  error */
              error(189, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
            if (A_TYPEG(tmp) != A_SUBSCR) {
              /* if the dummy is not assumed-shape
               * (explicit-shape or assumed-size), the user is
               * trying to pass a scalar or constant
               * to an array; give warning unless -Mstandard */
              if (ignore_tkr(arg, IGNORE_R))
                continue;
              if (DTY(eldact) == TY_CHAR || DTY(eldact) == TY_NCHAR)
                /*
                 * It's legal for a character scalar to be
                 * passed to a character array. This takes
                 * care of a scalar to an array but sill
                 * need to check types, POINTER, etc.
                 */
                ;
              else {
                if (flg.standard) {
                  error(189, 3, gbl.lineno, buf, SYMNAME(ext));
                  continue;
                }
                error(189, 2, gbl.lineno, buf, SYMNAME(ext));
                /*
                 * continue with checking types, POINTER, etc.
                 */
              }
            }
          }
        } else if (STYPEG(arg) == ST_PROC) {
          while (A_TYPEG(actual) == A_MEM) {
            actual = A_MEMG(actual);
          }
          if (A_TYPEG(actual) != A_ID) {
            error(447, 3, gbl.lineno, buf, SYMNAME(ext));
            continue;
          }
          sptr = A_SPTRG(actual);
          if (STYPEG(sptr) != ST_PROC && STYPEG(sptr) != ST_ENTRY &&
              STYPEG(sptr) != ST_INTRIN && STYPEG(sptr) != ST_GENERIC &&
              !(DTY(DTYPEG(sptr)) == TY_PTR &&
                DTY(DTY(DTYPEG(sptr) + 1)) == TY_PROC)) {
            error(447, 3, gbl.lineno, buf, SYMNAME(ext));
            continue;
          }
          /* FS#3742 Check that argument lists are compatible. */
          if (!compat_arg_lists(arg, sptr)) {
            char details[1000];
            sprintf(details, "- arguments of %s and %s do not agree",
                    SYMNAME(sptr), SYMNAME(arg));
            error(74, 3, gbl.lineno, SYMNAME(ext), details);
            continue;
          }
          if (ddum == 0) {
            /* formal has no dtype; was actual explicitly typed? */
            if (DCLDG(sptr) && DTYPEG(sptr) &&
                !(DTY(DTYPEG(sptr)) == TY_PTR &&
                  DTY(DTY(DTYPEG(sptr) + 1)) == TY_PROC)) {
              /* actual was given a datatype */
              error(448, 3, gbl.lineno, buf, SYMNAME(ext));
            }
            continue;
          }
          if (dact == 0) {
            /* actual has no datatype; was the formal explicitly typed? */
            if (DCLDG(arg)) { /* formal was declared */
              error(449, 3, gbl.lineno, buf, SYMNAME(ext));
            }
            continue;
          }
          if (!DCLDG(arg) && !FUNCG(arg) && !DCLDG(sptr) && !FUNCG(sptr))
            /* formal & actual are subroutines?? */
            continue;
        }
        if (DTY(ddum) == TY_ARRAY) {
          if (ASSUMSHPG(arg) && !ignore_tkr(arg, IGNORE_R)) {
            if (shape == 0) {
              error(189, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
            if (ADD_NUMDIM(ddum) != SHD_NDIM(shape)) {
              error(446, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
            if (!cmpat_arr_arg(ddum, shape)) {
              error(190, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
            if (SHD_UPB(shape, SHD_NDIM(shape) - 1) == 0) {
              error(191, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
          }
        } else if (is_iso_cloc(actual)) {

          /* smooth LOC() typechecking ? */
          A_DTYPEP(actual, elddum);
          eldact = elddum;
        } else if (!ELEMENTALG(ext) && DTY(dact) == TY_ARRAY) {
          /* scalar passed to array */
          if (!ignore_tkr(arg, IGNORE_R))
            error(446, 3, gbl.lineno, buf, SYMNAME(ext));
          continue;
        }
        /* Check if types of actual and dummy match.
         * When the the procedure argument is a procedure, the
         * type of the dummy may not be available.  When the actual
         * is an ST_PROC  and the dummy is an ST_IDENT, don't check
         * the case if the ST_PROC's FVAL field is not set.
         */
        if (A_TYPEG(actual) != A_ID ||
            (STYPEG(A_SPTRG(actual)) != ST_PROC || FVALG(A_SPTRG(actual))) ||
            STYPEG(arg) != ST_IDENT) {
          if (DTY(elddum) != DTY(eldact)) {
            if (eldact == 0 && STYPEG(sym_of_ast(actual)) == ST_PROC &&
                IS_PROC_DUMMYG(arg)) {
              continue;
            }
            if (DTY(elddum) == TY_DERIVED && UNLPOLYG(DTY(elddum + 3)))
              continue; /* FS#18004 */
            /* TY_ values are not the same */
            if (same_type_different_kind(elddum, eldact)) {
              /* kind differs */
              if (!ignore_tkr(arg, IGNORE_K))
                error(450, 3, gbl.lineno, buf, SYMNAME(ext));
              else {
                if (PASSBYVALG(arg) && DT_ISNUMERIC(DTYPEG(arg))) {
                  /*
                   * ensure the arg's semantic stack is
                   * evaluated and converted; this obviates
                   * the need for checking IGNORE_TKRG(
                   * param_dummy) in * semfunc.c:func_call2()
                   * - June 17,2015
                   */
                  (void)cngtyp(ARG_STK(i), DTYPEG(arg));
                  XFR_ARGAST(i);
                }
              }
            } else if (DTY(eldact) == TY_WORD) {
              if (A_TYPEG(actual) == A_INTR && A_OPTYPEG(actual) == I_NULL &&
                  A_ARGCNTG(actual) == 0 && POINTERG(arg)) {
                /* NULL() matches any POINTER formal */
                ;
              } else if (DT_ISWORD(elddum))
                ;
              else if (DT_ISDWORD(elddum)) {
                ARG_AST(i) = mk_convert(ARG_AST(i), DT_DWORD);
              } else {
                /* type differs */
                if (!ignore_tkr(arg, IGNORE_T))
                  error(188, 3, gbl.lineno, buf, SYMNAME(ext));
              }
            } else if (DTY(eldact) == TY_DWORD) {
              if (DT_ISDWORD(elddum))
                ;
              else if (DT_ISWORD(elddum)) {
                ARG_AST(i) = mk_convert(ARG_AST(i), DT_WORD);
              } else {
                /* type differs */
                if (!ignore_tkr(arg, IGNORE_T))
                  error(188, 3, gbl.lineno, buf, SYMNAME(ext));
              }
            } else {
              /* type differs */
              if (!ignore_tkr(arg, IGNORE_T))
                error(188, 3, gbl.lineno, buf, SYMNAME(ext));
              else if (!ignore_tkr(arg, IGNORE_K) &&
                       !different_type_same_kind(elddum, eldact))
                error(188, 3, gbl.lineno, buf, SYMNAME(ext));
            }
            continue;
          }
          /* check if type and kind of the data types match */
          if (!ignore_tkr(arg, IGNORE_T) &&
              !tk_match_arg(elddum, eldact, CLASSG(arg))) {
            if (DTY(elddum) != TY_DERIVED || !UNLPOLYG(DTY(elddum + 3))) {
              int mem;
              mem = get_generic_member(elddum, ext);
              if (!mem) {
                mem = get_generic_member(eldact, ext);
              }
              if (!mem || NOPASSG(mem)) {
                error(188, 3, gbl.lineno, buf, SYMNAME(ext));
                continue;
              }
              if (i == 0 && !PASSG(mem) &&
                  !tk_match_arg(eldact, elddum, CLASSG(arg))) {
                error(188, 3, gbl.lineno, buf, SYMNAME(ext));
                continue;
              }
              if (PASSG(mem) &&
                  strcmp(SYMNAME(PASSG(mem)), SYMNAME(arg)) == 0 &&
                  !tk_match_arg(eldact, elddum, CLASSG(arg))) {
                error(188, 3, gbl.lineno, buf, SYMNAME(ext));
                continue;
              }
            }
          }
        }
        if (POINTERG(arg)) {
          if (INTENTG(arg) != INTENT_IN) {
            int s;
            int iface;
            s = 0;
            switch (A_TYPEG(actual)) {
            case A_ID:
            case A_MEM:
              s = memsym_of_ast(actual);
              break;
            case A_FUNC:
              s = A_SPTRG(A_LOPG(actual));
              proc_arginfo(s, NULL, NULL, &iface);
              s = iface;
              break;
            case A_INTR:
              if (A_OPTYPEG(actual) == I_NULL)
                s = A_SPTRG(A_LOPG(actual));
              break;
            }
            if (s == 0 || (!POINTERG(s) &&
                           !(STYPEG(s) == ST_PD && PDNUMG(s) == PD_null))) {
              sprintf(buf, "%d (non-POINTER)", i + 1);
              error(188, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
          } else if (IS_CHAR_TYPE(DTYG(DTYPEG(arg)))) {
            if (!IS_CHAR_TYPE(DTYG(A_DTYPEG(actual)))) {
              error(188, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
            /* dummy must be adjustable length or the sizes must match */
            if (!ADJLENG(arg) && !eq_dtype(DTYPEG(arg), A_DTYPEG(actual))) {
              error(188, 3, gbl.lineno, buf, SYMNAME(ext));
              continue;
            }
          } else if (!eq_dtype2(DTYPEG(arg), A_DTYPEG(actual), TRUE)) {
            error(188, 3, gbl.lineno, buf, SYMNAME(ext));
            continue;
          } else if (CONTIGATTRG(arg) && !simply_contiguous(actual)) {
            error(546, 3, gbl.lineno, SYMNAME(arg), NULL);
            continue;
          }
        }
        if (ALLOCATTRG(arg)) {
          int s;
          int iface;
          s = 0;
          switch (A_TYPEG(actual)) {
          case A_ID:
            s = A_SPTRG(actual);
            break;
          case A_FUNC:
            s = A_SPTRG(A_LOPG(actual));
            proc_arginfo(s, NULL, NULL, &iface);
            s = iface;
            s = FVALG(s);
            break;
          case A_MEM:
            s = A_SPTRG(A_MEMG(actual));
            break;
          }
          if (s == 0 || !ALLOCATTRG(s)) {
            sprintf(buf, "%d (non-allocatable)", i + 1);
            error(188, 3, gbl.lineno, buf, SYMNAME(ext));
            continue;
          }
          NOALLOOPTP(s, 1);
        }
      }
      if (PASSBYVALG(arg)) {
        if (INTENTG(arg) == INTENT_OUT) {
          error(134, 3, gbl.lineno, "- INTENT(OUT) conflicts with VALUE",
                CNULL);
          continue;
        }
        if (INTENTG(arg) == INTENT_INOUT) {
          error(134, 3, gbl.lineno, "- INTENT(INOUT) conflicts with VALUE",
                CNULL);
          continue;
        }
      }
      if (actual && !A_ISLVAL(A_TYPEG(actual)) &&
          (INTENTG(arg) == INTENT_OUT || INTENTG(arg) == INTENT_INOUT)) {
        error(193, 2, gbl.lineno, buf, SYMNAME(ext));
        continue;
      }
      if (actual && A_ISLVAL(A_TYPEG(actual)) &&
          (INTENTG(arg) == INTENT_OUT || INTENTG(arg) == INTENT_INOUT)) {
        if (POINTERG(arg)) {
          /*
           * The formal argument is a pointer; the corresponding
           * actual cannot be an intent(in) argument.
           */
          sptr = find_pointer_variable(actual);
          if (sptr && POINTERG(sptr))
            (void)chk_pointer_intent(sptr, actual);
          if (is_protected(sym_of_ast(actual))) {
            err_protected(sptr, "be an actual argument when the dummy argument "
                                "is INTENT(OUT) or INTENT(INOUT)");
          }
        } else if (A_TYPEG(actual) == A_ID &&
                   SCG(A_SPTRG(actual)) == SC_DUMMY &&
                   INTENTG(A_SPTRG(actual)) == INTENT_IN &&
                   !POINTERG(A_SPTRG(actual)))
          error(193, 2, gbl.lineno, buf, SYMNAME(ext));
        sptr = sym_of_ast(actual);
        if (!POINTERG(sptr) && is_protected(sptr)) {
          err_protected(sptr, "be an actual argument when the dummy argument "
                              "is INTENT(OUT) or INTENT(INOUT)");
        }
      }
      if (!ALLOCATTRG(arg) && ASSUMSHPG(arg) && actual) {
        /* full descriptor required for 'arg' */
        int aid;
        switch (A_TYPEG(actual)) {
        case A_SUBSCR:
          aid = A_LOPG(actual);
          if (aid && A_TYPEG(aid) == A_ID) {
            sptr = A_SPTRG(aid);
            goto chk_allocatable;
          }
          break;
        case A_ID:
          sptr = A_SPTRG(actual);
          goto chk_allocatable;
        case A_MEM:
          sptr = A_SPTRG(A_MEMG(actual));
        chk_allocatable:
          if (ALLOCATTRG(sptr)) {
            ALLOCDESCP(sptr, TRUE);
          }
          FLANG_FALLTHROUGH;
        default:
          break;
        }
      }
    } else {
      if (actual == 0 || A_TYPEG(actual) != A_LABEL) {
        /* alternate returns */
        error(192, 3, gbl.lineno, buf, SYMNAME(ext));
      }
    }
  }

  return FALSE;
}

LOGICAL
ignore_tkr(int arg, int tkr)
{
  /*
   * Is it ok to ignore checking the specified TKR based on the presence
   * and value of the IGNORE_TKR directive?
   *
   * NOTE: * If we need to ignore the effects of the IGNORE_TKR directive,
   * guard the following 'if' with a test of an XBIT or whatever.
   */
  if (tkr == IGNORE_C && (IGNORE_TKRG(arg) & IGNORE_C) && ASSUMSHPG(arg))
    return TRUE;
  if ((IGNORE_TKRG(arg) & tkr) &&
      (ignore_tkr_all(arg) ||
      ((!ASSUMSHPG(arg) || ASSUMRANKG(arg))
      && !POINTERG(arg) && !ALLOCATTRG(arg))))
    return TRUE;
  return FALSE;
}

LOGICAL
ignore_tkr_all(int arg)
{
  if (((IGNORE_TKRG(arg) & IGNORE_TKR_ALL) == IGNORE_TKR_ALL) ||
      ((IGNORE_TKRG(arg) & IGNORE_TKR_ALL) == IGNORE_TKR_ALL0))
    return TRUE;
  return FALSE;
}

/** \brief Check conformance of two arrays, where the first array is described
 *         with a dtype record, and the second is described with a shape
 * descriptor.
 *
 *  Return true if the data types for two shapes are conformable (have the same
 *  shape). Shape is defined to be the rank and the extents of each dimension.
 */
static LOGICAL
cmpat_arr_arg(int d1, int shape2)
{
  int ndim;
  int i;
  int bnd;
  ADSC *ad1;
  INT lb1, lb2; /* lower bounds if constants */
  INT ub1, ub2; /* upper bounds if constants */
  INT st2;      /* stride of shape2 if constant */

  ad1 = AD_DPTR(d1);
  ndim = AD_NUMDIM(ad1);
  if (ndim != SHD_NDIM(shape2))
    return FALSE;

  for (i = 0; i < ndim; i++) {
    if ((bnd = AD_LWAST(ad1, i))) {
      if ((bnd = A_ALIASG(bnd)) == 0)
        continue; /* nonconstant bound => skip this dimension */
      lb1 = get_int_cval(A_SPTRG(bnd));
    } else
      lb1 = 1; /* no lower bound => 1 */

    if ((bnd = AD_UPAST(ad1, i))) {
      if ((bnd = A_ALIASG(bnd)) == 0)
        continue; /* nonconstant bound => skip this dimension */
      ub1 = get_int_cval(A_SPTRG(bnd));
    } else
      continue; /* no upper bound => skip this dimension */

    if ((lb2 = A_ALIASG(SHD_LWB(shape2, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    lb2 = get_int_cval(A_SPTRG(lb2));

    if ((ub2 = A_ALIASG(SHD_UPB(shape2, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    ub2 = get_int_cval(A_SPTRG(ub2));

    if ((st2 = A_ALIASG(SHD_STRIDE(shape2, i))) == 0)
      continue; /*  not a constant => skip this dimension */
    st2 = get_int_cval(A_SPTRG(st2));

    /* lower and upper bounds and stride are constants in this dimension*/

    if ((ub1 - lb1 + 1) != (ub2 - lb2 + st2) / st2)
      return FALSE;
  }

  return TRUE;
}

static int iface_arg(int, int, int);

int
iface_intrinsic(int sptr)
{
  int ii;
  int paramct, dtyper, argdtype;
  int ss;
  int iface, arg, dpdsc, fval;
  char *kwd, *np;
  int kwd_len;
  int optional;

  ii = INTASTG(sptr);
  switch (ii) {
  case I_ABS: /* abs */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ACOS: /* acos */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_AIMAG: /* aimag */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_AINT: /* aint */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ALOG: /* alog */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ALOG10: /* alog10 */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_AMOD: /* amod */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ANINT: /* anint */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ASIN: /* asin */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ATAN: /* atan */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_ATAN2: /* atan2 */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_CABS: /* cabs */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_CCOS: /* ccos */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_CEXP: /* cexp */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_CLOG: /* clog */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_CONJG: /* conjg */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_COS: /* cos */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_COSH: /* cosh */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_CSIN: /* csin */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_CSQRT: /* csqrt */
    paramct = 1;
    dtyper = DT_CMPLX;
    argdtype = DT_CMPLX;
    break;
  case I_DABS: /* dabs */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QABS: /* qabs */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
#endif
  case I_DACOS: /* dacos */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QACOS: /* qacos */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
#endif
  case I_DASIN: /* dasin */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DATAN: /* datan */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DATAN2: /* datan2 */
    paramct = 2;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QATAN2: /* qatan2 */
    paramct = 2;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
#endif
  case I_DCOS: /* dcos */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DCOSH: /* dcosh */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DDIM: /* ddim */
    paramct = 2;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DEXP: /* dexp */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DIM: /* dim */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_DINT: /* dint */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DLOG: /* dlog */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QLOG: /* qlog */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
#endif
  case I_DLOG10: /* dlog10 */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DMOD: /* dmod */
    paramct = 2;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DNINT: /* dnint */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DPROD: /* dprod */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_DSIGN: /* dsign */
    paramct = 2;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DSIN: /* dsin */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DSINH: /* dsinh */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DSQRT: /* dsqrt */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DTAN: /* dtan */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_DTANH: /* dtanh */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_EXP: /* exp */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QEXP: /* qexp */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QCOS: /* qcos */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QSIN: /* qsin */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QDIM: /* qdim */
    paramct = 2;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QATAN: /* qatan */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QTAN: /* qtan */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QASIN: /* qasin */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
  case I_QINT: /* qint */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
#endif
  case I_IABS: /* iabs */
    paramct = 1;
    dtyper = DT_INT;
    argdtype = DT_INT;
    break;
  case I_IDIM: /* idim */
    paramct = 2;
    dtyper = DT_INT;
    argdtype = DT_INT;
    break;
  case I_IDNINT: /* idnint */
    paramct = 1;
    dtyper = DT_DBLE;
    argdtype = DT_DBLE;
    break;
  case I_INDEX: /* index */
    paramct = 4;
    dtyper = DT_INT;
    argdtype = DT_ASSCHAR;
    break;
  case I_ISIGN: /* isign */
    paramct = 2;
    dtyper = DT_INT;
    argdtype = DT_INT;
    break;
  case I_LEN: /* len */
    paramct = 2;
    dtyper = DT_INT;
    argdtype = DT_ASSCHAR;
    break;
  case I_MOD: /* mod */
    paramct = 2;
    dtyper = DT_INT;
    argdtype = DT_INT;
    break;
  case I_NINT: /* nint */
    paramct = 2;
    dtyper = DT_INT;
    argdtype = DT_REAL;
    break;
  case I_SIGN: /* sign */
    paramct = 2;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_SIN: /* sin */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_SINH: /* sinh */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_SQRT: /* sqrt */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
#ifdef TARGET_SUPPORTS_QUADFP
  case I_QSQRT: /* qsqrt */
    paramct = 1;
    dtyper = DT_QUAD;
    argdtype = DT_QUAD;
    break;
#endif
  case I_TAN: /* tan */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  case I_TANH: /* tanh */
    paramct = 1;
    dtyper = DT_REAL;
    argdtype = DT_REAL;
    break;
  default:
    return 0;
  }
  ss = intast_sym[ii];
  /*
   * Build an 'abstract' interface from the intrinisic.
   * create a new symbol from the intrinsic with a fake name so that
   * it doesn't clash with a user or intrinsic symbol.
   */
  iface = getsymf("...%s", SYMNAME(ss));
  if (STYPEG(iface) != ST_UNKNOWN)
    return iface;
  CCSYMP(iface, 1);
  STYPEP(iface, ST_PROC);
  ABSTRACTP(iface, 1);
  DTYPEP(iface, dtyper);
  DCLDP(iface, 1);
  PUREP(iface, 1);
  if (INKINDG(ss) == IK_ELEMENTAL) {
    ELEMENTALP(iface, 1);
  }
  fval = iface_arg(iface, dtyper, iface);
  RESULTP(fval, 1);
  FVALP(iface, fval);
  FUNCP(iface, 1);
  PARAMCTP(iface, paramct);
  ++aux.dpdsc_avl; /* reserve one for fval */
  dpdsc = aux.dpdsc_avl;
  DPDSCP(iface, dpdsc);
  aux.dpdsc_avl += paramct;
  NEED(aux.dpdsc_avl, aux.dpdsc_base, int, aux.dpdsc_size,
       aux.dpdsc_size + paramct + 100);
  aux.dpdsc_base[dpdsc - 1] = fval;
  kwd = KWDARGSTR(ss);
  for (ii = 0; TRUE; ii++, kwd = np) {
    if (*kwd == '\0')
      break;
    if (*kwd == ' ')
      kwd++;
    if (*kwd != '*')
      optional = 0;
    else {
      optional = 1;
      kwd++;
    }
    kwd_len = 0;
    for (np = kwd; TRUE; np++) {
      if (*np == ' ' || *np == '\0')
        break;
      kwd_len++;
    }
    arg = getsym(kwd, kwd_len);
    arg = iface_arg(arg, argdtype, iface);
    OPTARGP(arg, optional);
    INTENTP(arg, INTENT_IN);
    aux.dpdsc_base[dpdsc] = arg;
    dpdsc++;
  }
#if DEBUG
  assert(ii == paramct, "iface_intrinsic: paramct does not match", iface, 3);
#endif
  return iface;
}

static int
iface_arg(int arg, int dt, int iface)
{
  if (STYPEG(arg) != ST_UNKNOWN)
    arg = insert_sym(arg);
  pop_sym(arg);
  STYPEP(arg, ST_VAR);
  DTYPEP(arg, dt);
  SCOPEP(arg, iface);
  SCP(arg, SC_DUMMY);
  DCLDP(arg, 1);
  NODESCP(arg, 1);
  IGNOREP(arg, 1);

  return arg;
}
