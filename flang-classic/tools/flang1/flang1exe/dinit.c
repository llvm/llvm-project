/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Process data initialization statements.  Called by semant.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "semant.h"
#include "semstk.h"
#include "dinit.h"
#include "ast.h"
#include "state.h"
#include "pd.h"

static int chk_doindex(int);
extern void dmp_ivl(VAR *, FILE *);
extern void dmp_ict(ACL *, FILE *);
static char *acl_idname(int);
static char *ac_opname(int);
static void dinit_data(VAR *, ACL *, int);
static ISZ_T arrayref_size(ACL *);
static void mark_dinit(VAR *, ACL *);
static void dinit_acl_val(int, int, ACL *);
static void dinit_intr_call(int, int, ACL *);
static void dinit_subs(ACL *);
static int dinit_val(int, int, int, int, int);
static int dinit_hollerith(int, int, int);
static void find_base(int, int *, int *);
static void sym_is_dinitd(int);

static LOGICAL no_dinitp = FALSE;

#define ERR170(s1, s2) error(170, 2, gbl.lineno, s1, s2)

/**
    Instead of creating dinit records during the processing of data
    initializations, we need to save information so the records are written
    at the end of semantic analysis (during semfin).  This is necessary for
    at least a couple of reasons: 1). a record dcl with inits in its STRUCTURE
    could occur before resolution of its storage class (problematic is
    SC_CMBLK)  2). with VMS ftn, an array may be initialized (not by implied
    DO) before resolution of its stype (i.e., its DIMENSION).

    The information we need to save is the pointers to the var list and
    constant tree.  This also implies that the getitem areas
    (4, 5) need to stay around until dinit output.
 */
void
dinit(VAR *ivl, ACL *ict)
{
  int nw;
  char *ptr;

  if (astb.df == NULL) {
    if ((astb.df = tmpfile()) == NULL)
      errfatal(5);
    sem.dinit_nbr_inits = 0;
  }
  nw = fwrite(&gbl.lineno, sizeof(gbl.lineno), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);
  nw = fwrite(&gbl.findex, sizeof(gbl.findex), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);
  ptr = (char *)ivl;
  nw = fwrite(&ptr, sizeof(ivl), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);
  ptr = (char *)ict;
  nw = fwrite(&ptr, sizeof(ict), 1, astb.df);
  if (nw != 1)
    error(10, 40, 0, "(data init file)", CNULL);

  if (ivl && ivl->u.varref.id == S_IDENT &&
      (STYPEG(A_SPTRG(ivl->u.varref.ptr)) == ST_PARAM ||
       PARAMG(A_SPTRG(ivl->u.varref.ptr)))) {
    sem.dinit_nbr_inits++;
  }
  mark_dinit(ivl, ict);
}

/** \brief Read in the information a "record" (1 word, 2 pointers) at a time
    saved by dinit(), and write dinit records for each record.
 */
void
do_dinit(void)
{
  VAR *ivl;
  ACL *ict;
  char *ptr;
  int nw;
  int fileno;

  if (astb.df == NULL)
    return;
  nw = fseek(astb.df, 0L, 0);
#if DEBUG
  assert(nw == 0, "do_dinit:bad rewind", nw, 4);
#endif

  while (TRUE) {
    nw = fread(&gbl.lineno, sizeof(gbl.lineno), 1, astb.df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "do_dinit: lineno error", nw, 4);
#endif
    /* Don't use gbl.findex here because we don't want its value to change */
    nw = fread(&fileno, sizeof(fileno), 1, astb.df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "do_dinit: fileno error", nw, 4);
#endif

    nw = fread(&ptr, sizeof(ivl), 1, astb.df);
    if (nw == 0)
      break;
#if DEBUG
    assert(nw == 1, "do_dinit: ict error", nw, 4);
#endif
    ivl = (VAR *)ptr;
    nw = fread(&ptr, sizeof(ict), 1, astb.df);
#if DEBUG
    assert(nw == 1, "do_dinit: ivl error", nw, 4);
#endif
    ict = (ACL *)ptr;
#if DEBUG
    if (DBGBIT(6, 32)) {
      fprintf(gbl.dbgfil, "---- deferred dinit read: ivl %p, ict %p\n",
              (void *)ivl, (void *)ict);
    }
#endif
    if (ict && ict->no_dinitp)
      no_dinitp = TRUE;
    df_dinit(ivl, ict);
    no_dinitp = FALSE;
  }

  if (gbl.maxsev >= 3) {
    /* since errors occur during semant, print_stmts() will not
     * be called; need to clean up the ast dinit file stuff.
     */
    fclose(astb.df);
    astb.df = NULL;
    /* freearea(15); */ /* saved dinit records & equivalence lists */
  }

}

void
dinit_no_dinitp(VAR *ivl, ACL *ict)
{
  no_dinitp = TRUE;
  ict->no_dinitp = 1;
  dinit(ivl, ict);
  no_dinitp = FALSE;
}

void
df_dinit_end()
{
  if (astb.df)
    fclose(astb.df);
  astb.df = NULL;
} /* df_dinit_end */

/**
    \param ivl pointer to initializer variable list
    \param ict pointer to initializer constant tree
 */
void
df_dinit(VAR *ivl, ACL *ict)
{
  if (DBGBIT(6, 3)) {
    fprintf(gbl.dbgfil, "\nDINIT CALLED ----------------\n");
    if (DBGBIT(6, 2)) {
      if (ivl) {
        fprintf(gbl.dbgfil, "  Dinit Variable List:\n");
        dmp_ivl(ivl, gbl.dbgfil);
      }
      if (ict) {
        fprintf(gbl.dbgfil, "  Dinit Constant List:\n");
        dmp_ict(ict, gbl.dbgfil);
      }
    }
    if (DBGBIT(6, 1))
      fprintf(gbl.dbgfil, "\n  DINIT Records:\n");
  }

  if (ivl) {
    sem.top = &sem.dostack[0];
    dinit_data(ivl, ict, 0); /* Process DATA statements */
  } else {
    sym_is_dinitd((int)ict->sptr);
    dinit_subs(ict); /* Process type dcl inits and */
  }                  /* init'ed structures */

  if (DBGBIT(6, 3))
    fprintf(gbl.dbgfil, "\nDINIT RETURNING ----------------\n\n");
}

static void
dinit_data(VAR *ivl, ACL *ict, int dtype)
{
  /* ivl : pointer to initializer variable list */
  /* ict : pointer to initializer constant tree */
  /* dtype : if this is a structure initialization, the ptr to dtype */

  int sptr, memptr;
  INT num_elem = 0;
  INT ict_rc = 0;
  LOGICAL is_array;
  int member = 0;
  int substr_dtype;

  if (ivl == NULL) {
    assert(dtype, "dinit_data: no object to initialize", 0, 2);
    member = DTY(dtype + 1);
    /* for derived type extension */
    if (PARENTG(DTY(dtype + 3)) && get_seen_contains()
    && (DTY(DTYPEG(member)) == TY_DERIVED)
    && (DTY(ict->dtype) != TY_DERIVED)) {
      member = SYMLKG(member);
    }
  }

  do {
    substr_dtype = 0;
    if (member && (is_empty_typedef(DTYPEG(member)) ||
                   is_tbp_or_final(member))) {
      memptr = SYMLKG(member);
      member = memptr == NOSYM ? 0 : memptr;
      continue;
    }
    if ((ivl && ivl->id == Varref) || member) {
      is_array = FALSE;
      num_elem = 1;
      if (member) {
        memptr = sptr = member;
        if (!POINTERG(sptr) && !ALLOCATTRG(sptr) &&
            DTY(DTYPEG(sptr)) == TY_ARRAY) {
          /* A whole array; determine number of elements to init */
          if (size_of_array(DTYPEG(sptr)))
            num_elem = get_int_cval(sym_of_ast(AD_NUMELM(AD_PTR(sptr))));
          else
            num_elem = 0;
          is_array = TRUE;
        }
      } else {
        int ast = ivl->u.varref.ptr;

        find_base(ast, &sptr, &memptr);
        if (sem.dinit_error)
          goto error_exit;
        if (A_TYPEG(ast) == A_ID || A_TYPEG(ast) == A_MEM) {
          /* We're initialising a scalar or whole array,
           * which may or may not be a derived type component.
           * (N.B. The derived type case may be an A_ID or
           * A_MEM node, depending on the value of DTF90OUTPUT.
           * In the former case, memptr == sptr.)
           */
          if (!POINTERG(sptr) && DTY(DTYPEG(memptr)) == TY_ARRAY) {
            /* A whole array */
            if (size_of_array(DTYPEG(memptr)))
              num_elem = get_int_cval(sym_of_ast(AD_NUMELM(AD_PTR(memptr))));
            else
              num_elem = 0;
            is_array = TRUE;
          }
        } else if (A_TYPEG(ast) == A_SUBSTR) {
          ISZ_T len;
          int s;
          s = A_SPTRG(A_ALIASG(A_RIGHTG(ast)));
          if (s)
            len = get_isz_cval(s);
          else
            len = string_length(DDTG(DTYPEG(memptr)));
          s = A_SPTRG(A_ALIASG(A_LEFTG(ast)));
          if (s)
            len = len - get_isz_cval(s) + 1;
          if (len < 0)
            len = 1;
          substr_dtype = get_type(2, DTY(A_DTYPEG(ast)), mk_cval(len, DT_INT4));
        } else {
          /* We're initialising an array element or section,
           */
          if (ivl->u.varref.shape != 0)
            uf("- initializing an array section");
        }
      }

      sym_is_dinitd(sptr);

      /*  now process enough dinit constant list items to
       *  take care of the current varref item.  For a Cray target,
       *  a Hollerith constant may initialize more than one array
       *  element.
       */
      do {
        if (ict_rc == 0) {
          if (ict == NULL) {
            if (is_array && XBIT(49, 0x1040000)) {
              /* T3D/T3E or C90 target: the number of initializers
               * may be less than the number of elements
               */
              if (flg.standard)
                ERR170("The number of initializers is less than number of "
                       "elements of",
                       SYMNAME(memptr));
              break;
            }
            errsev(66);
            goto error_exit;
          }
          ict_rc = dinit_eval(ict->repeatc);
        }
        if (ict_rc > 0) {
          /* Note: repeat factor ict_rc == 0 is allowed! */
          int ni; /* number of elements consumed by a constant */

          ni = 1;
          if (ivl && DTY(ivl->u.varref.dtype) == TY_DERIVED &&
              !is_zero_size_typedef(ivl->u.varref.dtype) && !POINTERG(sptr))
            dinit_data(ivl->u.varref.subt, ict->subc, ict->dtype);
          else if (member && DTY(ict->dtype) == TY_DERIVED &&
                   !is_zero_size_typedef(ict->dtype) && !POINTERG(sptr))
            if (ict->subc) {
              /* derived type member-by-member initialization */
              dinit_data(NULL, ict->subc, ict->dtype);
            } else {
              /* derived type initialized by a named constant */
              dinit_acl_val(member, ict->dtype, ict);
              return;
            }
          else if (is_array && ict->dtype == DT_HOLL)
            ni = dinit_hollerith(sptr, DDTG(DTYPEG(memptr)),
                                 A_SPTRG(A_ALIASG(ict->u1.ast)));
          else if (is_array) {
            if (ict->id == AC_IEXPR) {
              if (DTY(ict->dtype) == TY_ARRAY) {
                if (ict->u1.expr->op == AC_ARRAYREF) {
                  ni = arrayref_size(ict->u1.expr->rop);
                } else {
                  ni = dinit_eval(ADD_NUMELM(ict->dtype));
                }
                dinit_acl_val(sptr, DDTG(DTYPEG(memptr)), ict);
              } else {
                dinit_acl_val(sptr, DTYPEG(memptr), ict);
              }
            } else if (ict->id == AC_ACONST) {
              ACL *subict = ict->subc;
              /* MORE most of these calls to dinit_eval should be calls to
               * get_int_cval, dinit_eval is
               * for evaluating implied so expressions only */
              if (subict != 0 && subict->id == AC_IDO) {
                ni = num_elem; // Use the previously calculated size.
              } else {
                ni = dinit_eval(ADD_NUMELM(ict->dtype));
              }
              dinit_acl_val(sptr, DTYPEG(sptr), ict);
            } else {
              /* AC_AST, either a constant or a named constant */
              if (DTY(ict->dtype) == TY_ARRAY) {
                ni = dinit_eval(ADD_NUMELM(ict->dtype));
              }
              (void)dinit_val(sptr, DDTG(DTYPEG(memptr)), ict->dtype,
                              ict->u1.ast, 0);
            }
          } else if (substr_dtype) {
            dinit_acl_val(sptr, substr_dtype, ict);
          } else {
            /* Could be Superfluous ict if POINTER is set:
             * dinit_acl_val() will catch!
             */
            dinit_acl_val(sptr, DDTG(DTYPEG(memptr)), ict);
          }

          switch (ni) {
          case 1:
            ni = (ict_rc < num_elem) ? ict_rc : num_elem;
            num_elem -= ni;
            ict_rc -= ni;
            break;
          default:
            num_elem -= ni;
            ict_rc--;
            break;
          }
        }
        if (ict_rc == 0)
          ict = ict->next;
      } while (num_elem > 0);
      if (num_elem < 0)
        errsev(67);
    } else if (ivl->id == Dostart) {
      if (sem.top == &sem.dostack[MAX_DOSTACK]) {
        /*  nesting maximum exceeded.  */
        errsev(34);
        return;
      }
      sem.top->sptr = chk_doindex(ivl->u.dostart.indvar);
      if (sem.top->sptr == 1)
        return;
      sem.top->currval = dinit_eval(ivl->u.dostart.lowbd);
      sem.top->upbd = dinit_eval(ivl->u.dostart.upbd);
      sem.top->step = dinit_eval(ivl->u.dostart.step);
      if ((sem.top->step > 0 && sem.top->currval <= sem.top->upbd) ||
          (sem.top->step <= 0 && sem.top->currval >= sem.top->upbd))
        ++sem.top;
      else {
        /* A 'zero trip' implied DO loop.  Go directly to the
           corresponding 'Doend' node */
        int depth;

        depth = 1;
        do {
          ivl = ivl->next;
          if (!ivl)
            break;
          switch (ivl->id) {
          case Dostart:
            ++depth;
            break;
          case Doend:
            --depth;
            break;
          }
        } while (depth);
      }
    } else {
      assert(ivl->id == Doend, "dinit:badid", 0, 3);
      --sem.top;
      sem.top->currval += sem.top->step;
      if ((sem.top->step > 0 && sem.top->currval <= sem.top->upbd) ||
          (sem.top->step <= 0 && sem.top->currval >= sem.top->upbd)) {
        /*  go back to start of this do loop */
        ++sem.top;
        ivl = ivl->u.doend.dostart;
      }
    }
    if (sem.dinit_error)
      goto error_exit;
    if (ivl)
      ivl = ivl->next;
    if (member) {
      if (POINTERG(member) || ALLOCATTRG(member))
        member = SYMLKG(member); // skip <ptr>$p
      memptr = SYMLKG(member);
      member = memptr == NOSYM ? 0 : memptr;
    }
  } while (ivl || member);

  while (ict && num_elem > 0) {
    /* Some dinit constants remain.  That's OK if they have 0
     * repeat factor, otherwise it's an error. */
    if (ict_rc == 0)
      ict_rc = dinit_eval(ict->repeatc);
    if (ict_rc == 0)
      ict = ict->next;
    else {
      errsev(67);
      goto error_exit;
    }
  }
error_exit:
  return;
}

/*
 * compute the size of an arrayref (a section). The section information is
 * created by semant and consists of a list of ACL items of the form:
 * subscr_ast   <-(lop) [array_ref] (rop)-> lb -> ub -> stride, ...
 * repeated for the remaining dimensions.
 * Each ACL contains an const AST representing the value of the bounds/stride.
 */
static ISZ_T
arrayref_size(ACL *sect)
{
  ACL *c, *t;
  ISZ_T size, nelem;
  ISZ_T lowb, upb, stride;

  size = 1;
  t = sect;
  do {
    if (t == 0) {
      interr("dinit: arrayref: missing subscript array \n", 0, 3);
      return 1;
    }

    c = t->u1.expr->rop;

    if (c == 0) {
      interr("dinit: arrayref: missing array section lb\n", 0, 3);
      return 1;
    }
    lowb = get_isz_cval(A_SPTRG(c->u1.ast));

    if ((c = c->next) == 0) {
      interr("dinit: arrayref: missing array section ub\n", 0, 3);
      return 1;
    }
    upb = get_isz_cval(A_SPTRG(c->u1.ast));

    if ((c = c->next) == 0) {
      interr("dinit: arrayref: missing array section stride\n", 0, 3);
      return 1;
    }
    stride = get_isz_cval(A_SPTRG(c->u1.ast));

    if (stride < 0)
      nelem = (lowb - upb + (-stride)) / (-stride);
    else if (stride != 0)
      nelem = (upb - lowb + stride) / stride;
    else
      interr("dinit: arrayref: array section stride is 0\n", 0, 3);
    if (nelem > 0)
      size *= nelem;
    else
      size = 0;
    t = t->next;
  } while (t);
  return size;
}

/*---------------------------------------------------------------*/

/* pointer to initializer constant tree */
static void
dinit_subs(ACL *ict)
{
  int sptr; /* symbol ptr to identifier to get initialized */
  int i;

  /*
   * We come into this routine to follow the ict links for a substructure.
   */
  while (ict) {
    switch (ict->id) {
    case AC_TYPEINIT:
      dinit_subs(ict->subc);
      break;
    case AC_IDENT:
    case AC_CONST:
    case AC_IEXPR:
    case AC_AST:
    case AC_IDO:
    case AC_ACONST:
    case AC_SCONST:
    case AC_EXPR:
    case AC_REPEAT:
      dinit_acl_val(ict->sptr, DDTG(ict->dtype), ict);
      break;
    default:
      if (ict->subc) {
        /* Follow substructure down before continuing at this level */
        for (i = dinit_eval(ict->repeatc); i != 0; i--)
          dinit_subs(ict->subc);
      } else {
        /* Handle basic type declaration init statement */
        /* If new member or member has a repeat start a new block */
        if (ict->sptr) {
          /* A new member to initialize */
          sptr = ict->sptr;
        }
        (void)dinit_val(sptr, DDTG(DTYPEG(sptr)), ict->dtype, ict->u1.ast, 0);
      }
    }
    ict = ict->next;
  } /* End of while */
}

static void
setConval(int sptr, int conval, int op)
{
  if (conval && !PARMFING(sptr)) {
    int val = PARMINITG(sptr);
    switch (op) {
    case AC_ADD:
      val += conval;
      break;
    case AC_SUB:
      val -= conval;
      break;
    case AC_MUL:
      val *= conval;
      break;
    case AC_DIV:
      val /= conval;
      break;
    case AC_NEG:
      val = -conval;
      break;
    case AC_LNOT:
      val = ~conval;
      break;
    case AC_LOR:
      val |= conval;
      break;
    case AC_LAND:
      val &= conval;
      break;
    case AC_EQ:
      val = (val == conval) ? -1 : 0;
      break;
    case AC_GE:
      val = (val >= conval) ? -1 : 0;
      break;
    case AC_GT:
      val = (val > conval) ? -1 : 0;
      break;
    case AC_LE:
      val = (val <= conval) ? -1 : 0;
      break;
    case AC_LT:
      val = (val < conval) ? -1 : 0;
      break;
    case AC_NE:
      val = (val != conval) ? -1 : 0;
      break;
    case 0:
      val = conval;
      break;
    default:
      val = conval;
      error(155, 3, gbl.lineno, "Invalid operator for kind type parameter "
                                "initialization",
            NULL);
    }
    PARMINITP(sptr, val);
  }
}

static void
process_real_kind(int sptr, ACL *ict, int op)
{
  int ast, con1, conval;

  ast = ict->u1.ast;
  conval = 0;
  if (A_TYPEG(ast) == A_CNST) {

    con1 = A_SPTRG(ast);
    con1 = CONVAL2G(con1);
    if (con1 <= 6)
      conval = 4;
    else if (con1 <= 15)
      conval = 8;
    else if (con1 <= 31 && !XBIT(57, 4))
      conval = 16;
    else
      conval = -1;
  }

  ict = ict->next;
  if (ict) {
    ast = ict->u1.ast;

    if (A_TYPEG(ast) == A_CNST) {
      con1 = A_SPTRG(ast);
      con1 = CONVAL2G(con1);
      if (XBIT(49, 0x40000)) {
        /* Cray C90 */
        if (con1 <= 37) {
          if (conval > 0 && conval < 4)
            conval = 4;
        } else if (con1 <= 2465) {
          if (conval > 0 && conval < 8)
            conval = 8;
        } else {
          if (conval > 0)
            conval = 0;
          conval -= 2;
        }
      } else {
        /* ANSI */
        if (con1 <= 37) {
          if (conval > 0 && conval < 4)
            conval = 4;
        } else if (con1 <= 307) {
          if (conval > 0 && conval < 8)
            conval = 8;
        } else if (con1 <= 4931 && !XBIT(57, 4)) {
          if (conval > 0 && conval < 16)
            conval = 16;
        } else {
          if (conval > 0)
            conval = 0;
          conval -= 2;
        }
      }
    }
  }
  if (conval) {
    setConval(sptr, conval, op);
  }
}

static void
dinit_acl_val2(int sptr, int dtype, ACL *ict, int op)
{
  int dvl_val = 0;

  if (ict->id == AC_IEXPR) {
    switch (ict->u1.expr->op) {
    case AC_LNOT:
    case AC_NEG:
    case AC_CONV:
      dinit_acl_val2(sptr, dtype, ict->u1.expr->lop, 0);
      break;
    case AC_ADD:
    case AC_SUB:
    case AC_MUL:
    case AC_DIV:
    case AC_EXP:
    case AC_EXPK:
    case AC_EXPX:
    case AC_LOR:
    case AC_LAND:
    case AC_LEQV:
    case AC_LNEQV:
    case AC_EQ:
    case AC_GE:
    case AC_GT:
    case AC_LE:
    case AC_LT:
    case AC_NE:
      dinit_acl_val2(sptr, dtype, ict->u1.expr->lop, ict->u1.expr->op);
      dinit_acl_val2(sptr, dtype, ict->u1.expr->rop, ict->u1.expr->op);
      break;
    case AC_ARRAYREF:
      if (!cmpat_dtype_with_size(dtype, ict->dtype)) {
        errsev(91);
      }
      break;
    case AC_MEMBR_SEL:
      if (!cmpat_dtype_with_size(dtype, ict->dtype)) {
        errsev(91);
      }
      break;
    case AC_INTR_CALL:
      if (ict->id == AC_IEXPR) {
        ACL *subict = ict->u1.expr->rop;
        int intr = ict->u1.expr->lop->u1.i;
        int conval, con1, ast;
        if (subict && subict->id == AC_AST) {
          conval = 0;
          switch (intr) {
          case AC_I_selected_int_kind:
            ast = subict->u1.ast;
            if (A_TYPEG(ast) == A_CNST) {
              con1 = CONVAL2G(A_SPTRG(ast));
              conval = 4;
              if (con1 > 18 || (con1 > 9 && XBIT(57, 2)))
                conval = -1;
              else if (con1 > 9)
                conval = 8;
              else if (con1 > 4)
                conval = 4;
              else if (con1 > 2)
                conval = 2;
              else
                conval = 1;
            }
            setConval(sptr, conval, op);
            break;
          case AC_I_selected_real_kind:
            process_real_kind(sptr, subict, op);
            break;
          case AC_I_selected_char_kind:
            ast = subict->u1.ast;
            if (A_TYPEG(ast) == A_CNST) {
              int dty;
              con1 = A_SPTRG(ast);
              dty = DTY(DTYPEG(con1));
              if (dty == TY_CHAR || dty == TY_NCHAR) {
                conval = _selected_char_kind(con1);
              } else {
                interr("dinit: selected_char_kind: unexpected arg type", 0, 3);
                break;
              }
            }
            setConval(sptr, conval, op);
            break;
          default:
            /* Other intrinsics are handled by backend dinit. */
            break;
          }
        }
      }
      dinit_intr_call(sptr, dtype, ict);
      break;
    }
  } else if (ict->id == AC_ACONST) {
    if ((ict->subc != 0) && (ict->subc->id == AC_IDO)
        && (ict->subc->subc != 0) && (ict->subc->subc->id == AC_IDO)) {
      /* Perform a more relaxed check for a nested implied-do loop. */
      if (!cmpat_dtype_array_cast(dtype, ict->dtype)) {
        errsev(91);
      }
    } else {
      if (!cmpat_dtype_with_size(dtype, ict->dtype)) {
        errsev(91);
      }
      if (!cmpat_dtype_with_size(DDTG(dtype), DDTG(ict->dtype))) {
        errsev(91);
      }
    }
  } else if (ict->id == AC_IDO) {
    dinit_acl_val2(sptr, dtype, ict->subc, 0);
  } else if (ict->id == AC_AST) {
    /*
     * Superfluous ict if POINTER is set; would be better if we
     * didn't generate the entry, but ss a hack, just ignore it.
     */
    if (!POINTERG(sptr))
      dvl_val =
          dinit_val(sptr, dtype, DDTG(A_DTYPEG(ict->u1.ast)), ict->u1.ast, op);

    if (STYPEG(sptr) == ST_MEMBER && KINDG(sptr) && !USEKINDG(sptr) &&
        A_TYPEG(ict->u1.ast) == A_CNST) {
      int val = CONVAL2G(A_SPTRG(ict->u1.ast));
      setConval(sptr, val, op);
    }

  } else if (ict->id == AC_IDENT || ict->id == AC_CONST) {
    dvl_val =
        dinit_val(sptr, dtype, DDTG(A_DTYPEG(ict->u1.ast)), ict->u1.ast, op);
  }
  if (!XBIT(7, 0x100000)) {
    if (flg.opt >= 2 && sptr && STYPEG(sptr) == ST_VAR &&
        SCG(sptr) == SC_LOCAL && !ARGG(sptr) && !ASSNG(sptr) && dvl_val) {
      if (DTY(dtype) == TY_CHAR) {
        if (sptr && DTYPEG(sptr) != dtype)
          return;
        if (string_length(dtype) != string_length(DDTG(A_DTYPEG(ict->u1.ast))))
          return;
      } else if (DTY(dtype) == TY_NCHAR) {
        if (sptr && DTYPEG(sptr) != dtype)
          return;
        if (string_length(dtype) != string_length(DDTG(A_DTYPEG(ict->u1.ast))))
          return;
      }
      NEED(aux.dvl_avl + 1, aux.dvl_base, DVL, aux.dvl_size, aux.dvl_size + 32);
      DVL_SPTR(aux.dvl_avl) = sptr;
      DVL_CONVAL(aux.dvl_avl) = dvl_val;
      aux.dvl_avl++;
    }
  }
}

static void
dinit_acl_val(int sptr, int dtype, ACL *ict)
{

  dinit_acl_val2(sptr, dtype, ict, 0);
  if (STYPEG(sptr) == ST_MEMBER && KINDG(sptr) && !USEKINDG(sptr))
    PARMFINP(sptr, 1);
}

static void
dinit_intr_call(int sptr, int dtype, ACL *ict)
{
  ACL *aclp, *next_save;

  assert(ict->u1.expr->lop->id == AC_ICONST,
         "dinit_intr_call: incorrect ACL type for intrinsic selector\n", 0, 4);

  if (ict->u1.expr->lop->u1.i == AC_I_null) {
    /* Currently handles only NULL() */
    if (!POINTERG(sptr) && !PTRVG(sptr) && !ALLOCATTRG(sptr)) {
      errsev(459);
    }
    /* HACK: this is the only place before the backend where there is any
     * linkage between the VAR list and the ACL list (?).  Therefore it is
     * the only place where initialization of a <ptr> object with a NULL()
     * call can be modified.  First change the intrinsic call to a constant
     * zero initialization.  Then, if <ptr> is a derived type member, add
     * constant zeros to the ACL list to initialize any associated <ptr>$o,
     * <ptr>$sd, and/or <ptr>$td values that have been added as hidden
     * members of the type, but skip <ptr>$p values.  Processing for
     * non-derived type pointers is done in lower_pointer_init.
     */

    ict->id = AC_AST;
    ict->dtype = DT_PTR; /* may have problems with XBIT(125,0x2000) */
    if (!ict->ptrdtype) {
      /* build pointer type for backend upper/dinit */
      ict->ptrdtype = get_type(2, TY_PTR, DDTG(DTYPEG(sptr)));
    }
    if (ict->sptr && (POINTERG(ict->sptr) || ALLOCATTRG(sptr))) {
      /* use <ptr>$p */
      ict->sptr = MIDNUMG(sptr);
    }

    /* If astb.i0 will be changed to something else, it must change in
     * chk_struct_constructor as well.
     */
    ict->u1.ast = astb.i0;
    ict->is_const = 1;

    if (STYPEG(sptr) != ST_MEMBER)
      return;

    aclp = ict;
    next_save = ict->next;

    for (sptr = SYMLKG(SYMLKG(sptr)); HCCSYMG(sptr); sptr = SYMLKG(sptr)) {
      aclp = aclp->next = GET_ACL(15);
      aclp->id = AC_AST;
      aclp->is_const = 1;
      aclp->dtype = DDTG(DTYPEG(sptr));
      aclp->u1.ast = aclp->dtype == DT_INT8 ? astb.k0 : astb.i0;
      if (ict->sptr)
        aclp->sptr = sptr;
      if (DTY(DTYPEG(sptr)) == TY_ARRAY)
        aclp->repeatc = ADD_NUMELM(DTYPEG(sptr));
    }

    aclp->next = next_save;
  }
}

/*---------------------------------------------------------------*/

/* dinit_val - make sure constant value is correct data type to initialize
 * symbol (sptr) to.  Then call dinit_put to generate dinit record.
 */
static int
dinit_val(int sptr, int dtype, int dtypev, int astval, int op)
{
  INT val;
  INT newval[2];
  int newast;
  char buf[2];
  int do_dvl = 0;

  if (DTY(dtypev) == TY_DERIVED) {
    if (!eq_dtype(dtype, dtypev))
      errsev(91);
    return 0;
  }
  if (A_ALIASG(astval))
    astval = A_ALIASG(astval);

  if (is_procedure_ptr(sptr)) { 
    /* TBD: Eventually do this for regular pointers? */
    return astval;
  } 
  if (POINTERG(sptr)) {
    error(457, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    return 0;
  }
  if (!XBIT(7, 0x100000)) {
    if (flg.opt >= 2 && sptr && STYPEG(sptr) == ST_VAR &&
        SCG(sptr) == SC_LOCAL && !ARGG(sptr) && !ASSNG(sptr) &&
        DTY(DTYPEG(sptr)) != TY_DERIVED && op == 0) {
      do_dvl = 1;
    }
  }

  switch (DTY(A_DTYPEG(astval))) {
  case TY_DWORD:
  case TY_DBLE:
  case TY_QUAD:
  case TY_CMPLX:
  case TY_DCMPLX:
  case TY_QCMPLX:
  case TY_CHAR:
  case TY_NCHAR:
  case TY_INT8:
  case TY_LOG8:
    val = A_SPTRG(astval);
    break;
  case TY_ARRAY:
    /*
     * an array value does not require any special processing and
     * do not want to let it fall into the CHAR case; the CONVAL1
     * field of its sptr isn't defined.
     */
    return 0;
  case TY_HOLL:
    val = CONVAL1G(A_SPTRG(astval));
    break;
  default:
    val = CONVAL2G(A_SPTRG(astval));
  }

  if (DTYG(dtypev) == TY_HOLL) {
    /* convert hollerith character string to one of proper length */
    val = cngcon(val, (int)DTYPEG(val), dtype);
    if (do_dvl == 1) {
      switch (dtype) {
      case TY_DBLE:
      case TY_QUAD:
      case TY_INT8:
      case TY_LOG8:
      case TY_CMPLX:
      case TY_DCMPLX:
      case TY_QCMPLX:
        newast = mk_cnst(val);
        break;
      default:
        newval[0] = 0;
        newval[1] = val;
        val = getcon(newval, dtype);
        newast = mk_cnst(val);
      }
    }
  } else if (DTYG(dtypev) == TY_CHAR || DTYG(dtypev) == TY_NCHAR ||
             DTYG(dtypev) != DTY(dtype)) {
    /*  check for special case of initing character*1 to  numeric. */
    if (DTY(dtype) == TY_CHAR && DTY(dtype + 1) == astb.i1) {
      if (DT_ISINT(dtypev) && !DT_ISLOG(dtypev)) {
        if (flg.standard)
          error(172, 2, gbl.lineno, SYMNAME(sptr), CNULL);
        if (val < 0 || val > 255) {
          error(68, 3, gbl.lineno, SYMNAME(sptr), CNULL);
          val = getstring(" ", 1);
        } else {
          buf[0] = (char)val;
          buf[1] = 0;
          val = getstring(buf, 1);
        }
        dtypev = DT_CHAR;
      }
    }
    /* Convert character string to one of proper length or,
     * convert constant to type of identifier.
     */
    val = cngcon(val, dtypev, dtype);
    if (do_dvl == 1) {
      if (DTYG(dtypev) != DTY(dtype)) {
        switch (DTY(dtype)) {
        case TY_HOLL:
          val = getcon(&val, dtype);
          break;
        case TY_CHAR:
        case TY_NCHAR:
          break;
        case TY_LOG:
        case TY_SLOG:
        case TY_BLOG:
        case TY_INT:
        case TY_SINT:
        case TY_BINT:
        case TY_WORD:
        case TY_FLOAT:
          newval[0] = 0;
          newval[1] = val;
          val = getcon(newval, dtype);
          break;
        }
        newast = mk_cnst(val);
      } else {
        newast = mk_cnst(val);
      }
    }
  } else if (do_dvl == 1) {
    if (DTYG(dtypev) != DTY(dtype))
      newast = mk_cnst(val);
    else
      newast = astval;
  }

  if (do_dvl == 1 && op == 0) {
    return newast;
  }
  return 0;
}

/*
 * A Hollerith constant appears as a data item in the initialization of an
 * array.  For the certain targets (e.g., Cray), the constant may spill into
 * subsequent elements of the array.
 */
static int
dinit_hollerith(int sptr, int dtype, int holl_const)
{
  INT val;
  int ni; /* number of elements initialized by the constant */

  ni = 1; /* default number of initialized elements */

  val = CONVAL1G(holl_const); /* associated character constant */

  return ni;
}

/*---------------------------------------------------------------*/

/** \brief Dump an initializer variable list to a file (or stderr if no file
           provided).
 */
void
dmp_ivl(VAR *ivl, FILE *f)
{
  FILE *dfil;
  dfil = f ? f : stderr;
  while (ivl) {
    if (ivl->id == Dostart) {
      fprintf(dfil, "    Do begin marker  (%p):", (void *)ivl);
      fprintf(dfil, " indvar: %4d lowbd:%4d", ivl->u.dostart.indvar,
              ivl->u.dostart.lowbd);
      fprintf(dfil, " upbd:%4d  step:%4d\n", ivl->u.dostart.upbd,
              ivl->u.dostart.step);
    } else if (ivl->id == Varref) {
      if (ivl->u.varref.subt) {
        fprintf(dfil, " DERIVED TYPE members:\n");
        dmp_ivl(ivl->u.varref.subt, dfil);
        fprintf(dfil, " end DERIVED TYPE\n");
      } else {
        fprintf(dfil, "    Variable reference (");
        if (ivl->u.varref.id == S_IDENT) {
          fprintf(dfil, " S_IDENT):");
          fprintf(dfil, " sptr: %d(%s)", A_SPTRG(ivl->u.varref.ptr),
                  A_SPTRG(ivl->u.varref.ptr)
                      ? SYMNAME(A_SPTRG(ivl->u.varref.ptr))
                      : "");
          fprintf(dfil, " dtype: %4d\n", A_DTYPEG(ivl->u.varref.ptr + 1));
        } else {
          fprintf(dfil, "S_LVALUE):");
          fprintf(dfil, "  ast:%4d", ivl->u.varref.ptr);
          fprintf(dfil, " shape:%4d\n", ivl->u.varref.shape);
        }
      }
    } else {
      assert(ivl->id == Doend, "dmp_ivl: badid", 0, 3);
      fprintf(dfil, "    Do end marker:");
      fprintf(dfil, "   Pointer to Do Begin: %p\n",
              (void *)(ivl->u.doend.dostart));
    }
    ivl = ivl->next;
  }
}

/** \brief Dump an initializer constant tree to a file (dfil==0 --> stderr).
 */
void
dmp_ict(ACL *ict, FILE *dfil)
{
  static int level = 0;
  int i;

  if (!dfil)
    dfil = stderr;

  for (; ict; ict = ict->next) {
    for (i = level; i > 0; --i)
      fprintf(dfil, "  ");

    fprintf(dfil, "%p(%s):", (void *)ict, acl_idname(ict->id));
    if (ict->subc) {
      fprintf(dfil, "  subc:%p", ict->subc);
      if (ict->sptr) {
        fprintf(dfil, "  sptr:%d", ict->sptr);
        fprintf(dfil, "(%s)", SYMNAME(ict->sptr));
      }
      if (ict->repeatc)
        fprintf(dfil, "  rc:%d", ict->repeatc);
      fprintf(dfil, "  next:%p\n", (void *)(ict->next));
      ++level; dmp_ict(ict->subc, dfil);
    } else {
      if (ict->u1.ast)
        switch (ict->id) {
        case AC_EXPR:   fprintf(dfil, "  stkp:%p",   ict->u1.stkp);   break;
        case AC_IEXPR:  fprintf(dfil, "  expr:%p",   ict->u1.expr);   break;
        case AC_AST:
        case AC_CONST:
        case AC_IDENT:  fprintf(dfil, "  ast:%d",    ict->u1.ast);    break;
        case AC_ICONST: fprintf(dfil, "  iconst:%d", ict->u1.i);      break;
        case AC_REPEAT: fprintf(dfil, "  count:%d",  ict->u1.count);  break;
        case AC_IDO:    fprintf(dfil, "  doinfo:%p", ict->u1.doinfo); break;
        default:        fprintf(dfil, "  <u1>:%d",   ict->u1.i);
        }
      if (ict->dtype)
        fprintf(dfil, "  dtype:%d", ict->dtype);
      if (ict->repeatc)
        fprintf(dfil, "  rc:%d", ict->repeatc);
      if (ict->sptr) {
        fprintf(dfil, "  sptr:%d", ict->sptr);
        fprintf(dfil, "(%s)", SYMNAME(ict->sptr));
      }
      fprintf(dfil, "  next:%p\n", (void *)(ict->next));
    }

    if (ict->id == AC_IEXPR) {
      fprintf(dfil, "  lop:%p <OP %s> rop:%p\n", ict->u1.expr->lop,
              ac_opname(ict->u1.expr->op), ict->u1.expr->rop);
      ++level; dmp_ict(ict->u1.expr->lop, dfil);
      if (ict->u1.expr->rop) {
        ++level; dmp_ict(ict->u1.expr->rop, dfil);
      }
    }
  }

  if (level > 0)
    --level;
}

static char *
acl_idname(int id)
{
  static char bf[32];
  switch (id) {
  case AC_IDENT:
    strcpy(bf, "IDENT");
    break;
  case AC_CONST:
    strcpy(bf, "CONST");
    break;
  case AC_EXPR:
    strcpy(bf, "EXPR");
    break;
  case AC_IEXPR:
    strcpy(bf, "IEXPR");
    break;
  case AC_AST:
    strcpy(bf, "AST");
    break;
  case AC_IDO:
    strcpy(bf, "IDO");
    break;
  case AC_REPEAT:
    strcpy(bf, "REPEAT");
    break;
  case AC_ACONST:
    strcpy(bf, "ACONST");
    break;
  case AC_SCONST:
    strcpy(bf, "SCONST");
    break;
  case AC_LIST:
    strcpy(bf, "LIST");
    break;
  case AC_VMSSTRUCT:
    strcpy(bf, "VMSSTRUCT");
    break;
  case AC_VMSUNION:
    strcpy(bf, "VMSUNION");
    break;
  case AC_TYPEINIT:
    strcpy(bf, "TYPEINIT");
    break;
  case AC_ICONST:
    strcpy(bf, "ICONST");
    break;
  case AC_CONVAL:
    strcpy(bf, "CONVAL");
    break;
  case AC_TRIPLE:
    strcpy(bf, "TRIPLE");
    break;
  default:
    sprintf(bf, "UNK_%d", id);
    break;
  }
  return bf;
}

static char *
ac_opname(int id)
{
  static char bf[32];
  switch (id) {
  case AC_ADD:
    strcpy(bf, "ADD");
    break;
  case AC_SUB:
    strcpy(bf, "SUB");
    break;
  case AC_MUL:
    strcpy(bf, "MUL");
    break;
  case AC_DIV:
    strcpy(bf, "DIV");
    break;
  case AC_NEG:
    strcpy(bf, "NEG");
    break;
  case AC_EXP:
    strcpy(bf, "EXP");
    break;
  case AC_INTR_CALL:
    strcpy(bf, "INTR_CALL");
    break;
  case AC_ARRAYREF:
    strcpy(bf, "ARRAYREF");
    break;
  case AC_MEMBR_SEL:
    strcpy(bf, "MEMBR_SEL");
    break;
  case AC_CONV:
    strcpy(bf, "CONV");
    break;
  case AC_CAT:
    strcpy(bf, "CAT");
    break;
  case AC_EXPK:
    strcpy(bf, "EXPK");
    break;
  case AC_LEQV:
    strcpy(bf, "LEQV");
    break;
  case AC_LNEQV:
    strcpy(bf, "LNEQV");
    break;
  case AC_LOR:
    strcpy(bf, "LOR");
    break;
  case AC_LAND:
    strcpy(bf, "LAND");
    break;
  case AC_EQ:
    strcpy(bf, "EQ");
    break;
  case AC_GE:
    strcpy(bf, "GE");
    break;
  case AC_GT:
    strcpy(bf, "GT");
    break;
  case AC_LE:
    strcpy(bf, "LE");
    break;
  case AC_LT:
    strcpy(bf, "LT");
    break;
  case AC_NE:
    strcpy(bf, "NE");
    break;
  case AC_LNOT:
    strcpy(bf, "LNOT");
    break;
  case AC_EXPX:
    strcpy(bf, "EXPX");
    break;
  case AC_TRIPLE:
    strcpy(bf, "TRIPLE");
    break;
  default:
    sprintf(bf, "ac_opnameUNK_%d", id);
    break;
  }
  return bf;
}

/*---------------------------------------------------------------*/

/* find_base - dereference an ast pointer to determine the base
 *             of an array reference (i.e. base sptr).
 */
static void
find_base(int ast, int *psptr, int *pmemptr)
{
  int sptr, memptr = 0, a;
  int i;
  int asd;
  ADSC *ad;
  int ndim;
  int lwbd;
  int offset;

  switch (A_TYPEG(ast)) {
  case A_SUBSTR:
    find_base((int)A_LOPG(ast), &sptr, &memptr);
    if (sem.dinit_error)
      break;
    /* check left & right indices */
    (void)dinit_eval((int)A_LEFTG(ast));
    (void)dinit_eval((int)A_RIGHTG(ast));
    break;

  case A_SUBSCR:
    find_base((int)A_LOPG(ast), &sptr, &memptr);
    if (sem.dinit_error)
      break;
    asd = A_ASDG(ast);
    ad = AD_PTR(memptr);
    ndim = ASD_NDIM(asd);
    for (i = 0; i < ndim; i++) {
      lwbd = get_int_cval(sym_of_ast(AD_LWAST(ad, i)));
      offset = dinit_eval((int)ASD_SUBS(asd, i));
      if (offset < lwbd || offset > get_int_cval(sym_of_ast(AD_UPAST(ad, i)))) {
        error(80, 3, gbl.lineno, SYMNAME(sptr), CNULL);
        sem.dinit_error = TRUE;
        break;
      }
    }
    break;

  case A_ID:
    if (A_ALIASG(ast))
      goto err;
    memptr = sptr = A_SPTRG(ast);
    (void)dinit_ok(sptr);
    break;

  case A_MEM:
    a = A_PARENTG(ast);
    if (A_TYPEG(a) == A_SUBSCR)
      a = A_LOPG(a);
    sptr = A_SPTRG(a);
    a = A_MEMG(ast);
    memptr = A_SPTRG(a);
    break;

  case A_FUNC:
    sptr = A_LOPG(ast);
    error(76, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    sem.dinit_error = TRUE;
    break;

  default:
  err:
    memptr = sptr = 0;
    sem.dinit_error = TRUE;
    break;
  }
  *psptr = sptr;
  *pmemptr = memptr;
}

/*---------------------------------------------------------------*/

/*
 * find the sptr for the implied do index variable; the ilm in this
 * context represents the ilms generated to load the index variable
 * and perhaps "type" convert (if it's integer*2, etc.).
 */
static int
chk_doindex(int ast)
{
again:
  switch (A_TYPEG(ast)) {
  case A_CONV:
    ast = A_LOPG(ast);
    goto again;
  case A_ID:
    if (!DT_ISINT(A_DTYPEG(ast)) || A_ALIASG(ast))
      break;
    return A_SPTRG(ast);
  }
  /* could use a better error message - illegal implied do index variable */
  errsev(106);
  sem.dinit_error = TRUE;
  return 1L;
}

INT
dinit_eval(int ast)
{
  DOSTACK *p;
  int sptr;

  if (ast == 0)
    return 1L;
  if (!DT_ISINT(A_DTYPEG(ast)))
    goto err;
  if (A_ALIASG(ast)) {
    ast = A_ALIASG(ast);
    goto eval_cnst;
  }
  switch (A_TYPEG(ast) /* opc */) {
  case A_ID:
    if (!DT_ISINT(A_DTYPEG(ast)))
      goto err;
    if (A_ALIASG(ast)) {
      ast = A_ALIASG(ast);
      goto eval_cnst;
    }
    /*  see if this ident is an active do index variable: */
    sptr = A_SPTRG(ast);
    for (p = sem.dostack; p < sem.top; p++)
      if (p->sptr == sptr)
        return p->currval;
    /*  else - illegal use of variable: */
    error(64, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    sem.dinit_error = TRUE;
    return 1L;

  case A_CNST:
    goto eval_cnst;

  case A_UNOP:
    if (A_OPTYPEG(ast) == OP_SUB)
      return -dinit_eval((int)A_LOPG(ast));
    return dinit_eval((int)A_LOPG(ast));
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      return dinit_eval((int)A_LOPG(ast)) + dinit_eval((int)A_ROPG(ast));
    case OP_SUB:
      return dinit_eval((int)A_LOPG(ast)) - dinit_eval((int)A_ROPG(ast));
    case OP_MUL:
      return dinit_eval((int)A_LOPG(ast)) * dinit_eval((int)A_ROPG(ast));
    case OP_DIV:
      return dinit_eval((int)A_LOPG(ast)) / dinit_eval((int)A_ROPG(ast));
    }
    break;
  case A_CONV:
  case A_PAREN:
    return dinit_eval((int)A_LOPG(ast));

  case A_INTR:
    if (A_OPTYPEG(ast) == I_NULL) {
      return 0;
    }
    FLANG_FALLTHROUGH;
  default:
    break;
  }
err:
  errsev(69);
  sem.dinit_error = TRUE;
  return 1L;
eval_cnst:
  return get_int_cval(A_SPTRG(ast));
}

/*---------------------------------------------------------------*/

/*
 * sym_is_dinitd: a symbol is being initialized - update certain
 * attributes of the symbol including its dinit flag.
 */
static void
sym_is_dinitd(int sptr)
{
  if (no_dinitp)
    return;
  DINITP(sptr, 1);
  if (is_procedure_ptr(sptr)) { 
    /* TBD: Eventually do this for regular pointers? */
    SPTR ptr, sdsc, off;
    ptr = MIDNUMG(sptr);
    DINITP(MIDNUMG(sptr), 1);
    sdsc = SDSCG(sptr);
    if (sdsc && STYPEG(sdsc) != ST_PARAM) {
      DINITP(sdsc, 1);
     }
     off = PTROFFG(sptr);
     if (off && STYPEG(off) != ST_PARAM) {
       DINITP(off, 1);
     }
  } 
  if (ST_ISVAR(STYPEG(sptr)) && SCG(sptr) == SC_CMBLK)
    /*  set DINIT flag for common block:  */
    DINITP(CMBLKG(sptr), 1);

  /* For identifiers the DATA statement ensures that the identifier
   * is a variable and not an intrinsic.  For arrays, either
   * compute the element offset or if a whole array reference
   * compute the number of elements to initialize.
   */
  if (STYPEG(sptr) == ST_IDENT || STYPEG(sptr) == ST_UNKNOWN)
    STYPEP(sptr, ST_VAR);

}

static void
mark_ivl_dinit(VAR *ivl)
{
  while (ivl != NULL && ivl->id == Varref) {
    if (ivl->u.varref.subt) {
      mark_ivl_dinit(ivl->u.varref.subt);
    } else {
      int sptr;
      sptr = sym_of_ast(ivl->u.varref.ptr);
      sym_is_dinitd(sptr);
    }
    ivl = ivl->next;
  }
} /* mark_ivl_dinit */

static void
mark_dinit(VAR *ivl, ACL *ict)
{
  if (ivl == NULL) {
    sym_is_dinitd(ict->sptr);
  } else {
    mark_ivl_dinit(ivl);
  }
} /* mark_dinit */

/*---------------------------------------------------------------*/

/*  determine if the symbol can be legally data initialized  */
LOGICAL
dinit_ok(int sptr)
{
  switch (SCG(sptr)) {
  case SC_DUMMY:
    error(41, 3, gbl.lineno, SYMNAME(sptr), CNULL);
    goto error_exit;
  case SC_CMBLK:
    if (ALLOCG(MIDNUMG(sptr))) {
      error(163, 3, gbl.lineno, SYMNAME(sptr), SYMNAME(MIDNUMG(sptr)));
      goto error_exit;
    }
    break;
  default:
    break;
  }
  if (STYPEG(sptr) == ST_ARRAY && !POINTERG(sptr)) {
    if (ALLOCG(sptr)) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- initializing an allocatable array");
      goto error_exit;
    }
    if (ASUMSZG(sptr)) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- initializing an assumed size array");
      goto error_exit;
    }
    if (ADJARRG(sptr)) {
      error(84, 3, gbl.lineno, SYMNAME(sptr),
            "- initializing an adjustable array");
      goto error_exit;
    }
  }
  if (ADJLENG(sptr)) {
    error(84, 3, gbl.lineno, SYMNAME(sptr),
          "- initializing an adjustable length object");
    goto error_exit;
  }

  return TRUE;

error_exit:
  sem.dinit_error = TRUE;
  return FALSE;
}

void rw_dinit_state(RW_ROUTINE, RW_FILE)
{

  VAR *ivl;
  ACL *ict;
  int nw;
  int lineno;
  FILE *readfile;
  FILE *writefile;
  int i;
  int seq_astb_df;
  int fileno = 1;

  seq_astb_df = 0;
  if (ISREAD()) {
    if (astb.df == NULL) {
      if ((astb.df = tmpfile()) == NULL)
        errfatal(5);
    } else {
      nw = fseek(astb.df, 0L, 0);
#if DEBUG
      assert(nw == 0, "do_dinit:bad rewind", nw, 4);
#endif
    }

    /* restore, read saved state and write dinit file */
    readfile = fd; /* from parameter RW_FILE */
    writefile = astb.df;
  } else {
    if (astb.df == NULL) {
      /* this can happen if there are errors */
      sem.dinit_nbr_inits = 0;
      RW_SCALAR(sem.dinit_nbr_inits);
      return;
    }
    nw = fseek(astb.df, 0L, 0);
#if DEBUG
    assert(nw == 0, "do_dinit:bad rewind", nw, 4);
#endif
    /* save, read dinit file and write saved state */
    readfile = astb.df;
    seq_astb_df = 1;
    writefile = fd; /* from parameter RW_FILE */
  }

  RW_SCALAR(sem.dinit_nbr_inits);

  for (i = sem.dinit_nbr_inits; i;) {
    nw = fread(&lineno, sizeof(lineno), 1, readfile);
    if (nw != 1)
      break;

    nw = fread(&fileno, sizeof(fileno), 1, readfile);
    if (nw != 1)
      break;

    nw = fread(&ivl, sizeof(VAR *), 1, readfile);
    if (nw != 1)
      break;

    nw = fread(&ict, sizeof(ACL *), 1, readfile);
    if (nw != 1)
      break;

    /* save/restore only parameter initializations */
    if (!ivl || ivl->u.varref.id != S_IDENT ||
        (STYPEG(A_SPTRG(ivl->u.varref.ptr)) != ST_PARAM &&
         !PARAMG(A_SPTRG(ivl->u.varref.ptr)))) {
      continue;
    }

    nw = fwrite(&lineno, sizeof(lineno), 1, writefile);
    if (nw != 1)
      break;

    nw = fwrite(&fileno, sizeof(fileno), 1, writefile);
    if (nw != 1)
      break;

    nw = fwrite(&ivl, sizeof(VAR *), 1, writefile);
    if (nw != 1)
      break;

    nw = fwrite(&ict, sizeof(ACL *), 1, writefile);
    if (nw != 1)
      break;

    i--;
  }

  if (i != 0) {
    interr("dinit save/restore failed", 0, 4);
  }

  if (seq_astb_df) {
    /*
     * If the next I/O operation on astb.df is a write, the write will
     * fail on win.  Strictly speaking, a file positioning operation
     * must be performed before the write.  This was the cause of
     * "data init file" write errors when compiling relatively simple
     * f90 programs; all that's needed to be present is dinits in
     * modules or host subprograms and contained subprograms.
     */
    long file_pos;
    file_pos = ftell(astb.df);
    (void)fseek(astb.df, file_pos, 0);
  }
}
