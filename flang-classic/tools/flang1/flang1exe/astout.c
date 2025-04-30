/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Abstract syntax tree output module.
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "symutl.h"
#include "dtypeutl.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "pragma.h"
#include "gramtk.h"
#include "tokdf.h"
#include "dinit.h"
#include "rte.h"
#include "rtlRtns.h"

#define NO_PTR XBIT(49, 0x8000)
#define NO_CHARPTR XBIT(58, 0x1)
#define NO_DERIVEDPTR XBIT(58, 0x40000)

/* The only routine that writes to 'outfile' is write_next_line */
static FILE *outfile;
static int col = 0;
static int max_col = 72;

static int continuations = 0; /* number of continuation lines */

static int indent; /* number of indentation levels */

#define CARDB_SIZE 2100 /* make it large enough */
static char lbuff[CARDB_SIZE];

#define MAX_FNAME_LEN 2050
static LOGICAL ast_is_comment = FALSE;
static LOGICAL op_space = TRUE;

static LOGICAL altret_spec = FALSE; /* labels are alternate return specifiers */

typedef struct { /* simple queue decl. */
  int first;
  int last;
} _A_Q;

/* create queue of symbols specified in parameter statements; keep
 * separate queues for combinations of ansi-/vax- style parameters
 * and those with A_CNST/non-A_CNST asts.
 * 'first' locates first in the queue and is 0 if the queue is empty;
 * symbols are linked together using the SYMLK field; queue is terminated
 * when the SYMLK field is zero. 'last' locates the last parameter in
 * the queue.
 */
static struct {
  _A_Q q;   /* queue for parameters with const ast's */
  _A_Q q_e; /* queue for parameters with expr ast's */
} params, vx_params = {0};

typedef struct _qsym { /* for queuing syms whose decls are to be printed later*/
  struct _qsym *next;
  int sptr;
} QSYM;

static void print_ast(int ast); /* fwd decl */
static void print_ast_replaced(int, int, int);

static void init_line(void);
static void push_indent(void);
static void pop_indent(void);
static void print_uncoerced_const(int);
static void print_loc(int);
static void print_loc_of_sym(int);
static void print_refsym(int, int);
static void print_sname(int);
static void print_naked_id(int);
void deferred_to_pointer(void);
static int pr_chk_arr(int);
static void gen_bnd_assn(int);
static void gen_allocate(int, int);
static void gen_deallocate(int, int, int, int);
static void gen_nullify(int, int, int);
static void put_mem_string(int, const char *);
static void put_string(const char *);
static void put_fstring(const char *);
static void put_char(char);
static void put_const(int);
static void put_int(INT);
static void put_intkind(INT, int);
static void put_int8(int);
static void put_logical(LOGICAL, int);
static void put_float(INT);
static void put_double(int);
static void char_to_text(int);
static void put_u_to_l(const char *);
static void put_l_to_u(const char *);
static void check_len(int);
static char *label_name(int);
static void print_header(int);
static void pghpf_entry(int);
static void put_call(int ast, int call, const char *name, int check_ptrarg);

void
astout_init(void)
{
  if (XBIT(52, 0x20))
    max_col = 132;
  BZERO(&params, char, sizeof(params));
  BZERO(&vx_params, char, sizeof(vx_params));
}

static void
init_line(void)
{
  col = 0;
  put_string("      "); /* 6 blanks */
}

#define INDENT_MAX 4
#define INDENT_STR "   "

static void
push_indent(void)
{
  if (!ast_is_comment) {
    indent++;
    if (indent <= INDENT_MAX)
      put_string(INDENT_STR);
  }
}

static void
pop_indent(void)
{
  if (!ast_is_comment) {
    indent--;
    if (indent < 0) {
      interr("pop_indent:ident_level", indent, ERR_Warning);
      indent = 0;
    }
    if (indent < INDENT_MAX)
      col -= strlen(INDENT_STR);
  }
}

static int
precedence(int ast)
{
/*
 * Precedence Levels:
 * 20   identifiers, function calls, parens, etc.; any 'term'
 * 18   **
 * 16   * /
 * 14   + - (binary)
 * 12   + - (unary)
 * 10   relationals
 *  8   .not.
 *  6   .and.
 *  4   .or.
 *  2   .neqv. .eqv.
 */
#define PREC_TERM 20
#define PREC_POW 18
#define PREC_MULT 16
#define PREC_ADD 14
#define PREC_NEG 12
#define PREC_REL 10
#define PREC_NOT 8
#define PREC_AND 6
#define PREC_OR 4
#define PREC_EQV 2

  switch (A_TYPEG(ast)) {
  case A_BINOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
      return PREC_ADD;
    case OP_MUL:
    case OP_DIV:
      return PREC_MULT;
    case OP_XTOI:
    case OP_XTOX:
      return PREC_POW;
    case OP_CAT:
      return PREC_MULT;
    case OP_LEQV:
    case OP_LNEQV:
      return PREC_EQV;
    case OP_LOR:
      return PREC_OR;
    case OP_LAND:
    case OP_SCAND:
      return PREC_AND;
    case OP_EQ:
    case OP_GE:
    case OP_GT:
    case OP_LE:
    case OP_LT:
    case OP_NE:
      return PREC_REL;
    default:
      break;
    }
    break;
  case A_UNOP:
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
    case OP_SUB:
      return /* PREC_NEG */ PREC_ADD;
    case OP_LNOT:
      return PREC_NOT;
    case OP_LOC:
    case OP_REF:
    case OP_VAL:
    case OP_BYVAL:
      break;
    default:
      break;
    }
    break;
  case A_CONV:
    return precedence((int)A_LOPG(ast));
  default:
    break;
  }
  return PREC_TERM;
}

static LOGICAL
negative_constant(int ast)
{
  DBLINT64 inum1, inum2;
  DBLE dnum1, dnum2;

  if (A_TYPEG(ast) == A_CNST) {
    int sptr;
    sptr = A_SPTRG(ast);
    switch (DTY(DTYPEG(sptr))) {
    case TY_INT:
      if (CONVAL2G(sptr) & 0x80000000)
        return TRUE;
      break;
    case TY_REAL:
      if (xfcmp(CONVAL2G(sptr), CONVAL2G(stb.flt0)) < 0)
        return TRUE;
      break;
    case TY_DBLE:
      dnum1[0] = CONVAL1G(sptr);
      dnum1[1] = CONVAL2G(sptr);
      dnum2[0] = CONVAL1G(stb.dbl0);
      dnum2[1] = CONVAL2G(stb.dbl0);
      if (xdcmp(dnum1, dnum2) < 0)
        return TRUE;
      break;
    case TY_INT8:
      inum1[0] = CONVAL1G(sptr);
      inum1[1] = CONVAL2G(sptr);
      inum2[0] = 0;
      inum2[1] = 0;
      if (cmp64(inum1, inum2) < 0)
        return TRUE;
      break;
    default:
      break;
    }
  }
  return FALSE;
}

static int
left_precedence(int lop, int prec_op)
{
  int prec_lop;
  while (A_TYPEG(lop) == A_CONV)
    lop = A_LOPG(lop);
  if (negative_constant(lop))
    /*
     * a constant represents the highest precedence level since it's
     * a term. Treating it as a term is a problem if a negative constant
     * is the left operand of a binary operator; the precedence needs to
     * be the precedence of a unary minus.
     */
    return PREC_ADD;

  prec_lop = precedence(lop);
  if (prec_op == PREC_POW && prec_lop == PREC_POW)
    /* left operand of ** is also a **; since  '**' is right
     * associative, need to ensure that the left operand is
     * parenthesized.
     */
    prec_lop--;
  return prec_lop;
}

static int
right_precedence(int rop, int prec_op)
{
  int prec_rop;

  while (A_TYPEG(rop) == A_CONV)
    rop = A_LOPG(rop);
  if (negative_constant(rop))
    /*
     * a constant represents the highest precedence level since it's
     * a term. Treating it as a term is a problem if a negative constant
     * is the right operand of a binary operator; the precedence needs to
     * be the precedence of a unary minus.
     */
    return PREC_ADD;

  prec_rop = precedence(rop);
  if (prec_op == PREC_POW && prec_rop == PREC_POW)
    /* right operand of ** is also a **; since  '**' is right
     * associative, need to ensure that the right operand is
     * not parenthesized.
     */
    prec_rop++;
  return prec_rop;
}

static void
cuf_pragma(int ast)
{
  lbuff[0] = '!';
  lbuff[1] = '$';
  lbuff[2] = 'c';
  lbuff[3] = 'u';
  lbuff[4] = 'f';
  lbuff[5] = ' ';
} /* cuf_pragma */

static void
acc_pragma(int ast)
{
  lbuff[0] = '!';
  lbuff[1] = '$';
  lbuff[2] = 'a';
  lbuff[3] = 'c';
  lbuff[4] = 'c';
  lbuff[5] = ' ';
} /* acc_pragma */

/* device type */
static void
acc_dtype(int ast)
{
} /* acc_dtype */

static void
print_ast(int ast)
{
  const char *o;
  int atype;
  int i, asd;
  int astli;
  int argt;
  int cnt;
  int lop, rop;
  int prec_op, prec_lop;
  int shape;
  LOGICAL encl;
  int linearize;
  LOGICAL save_op_space;
  LOGICAL commutable, nid;
  FtnRtlEnum rtlRtn;
  int sym, object;
  int dtype;
  int optype;
  int save_dtype, save_comment;

  switch (atype = A_TYPEG(ast)) {
  case A_NULL:
    break;
  case A_ID:
    print_refsym(A_SPTRG(ast), ast);
    break;
  case A_CNST:
    put_const((int)A_SPTRG(ast));
    break;
  case A_LABEL:
    if (altret_spec)
      put_char('*');
    put_string(label_name((int)A_SPTRG(ast)));
    break;
  case A_BINOP:
    lop = A_LOPG(ast);
    rop = A_ROPG(ast);
    commutable = FALSE;
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      o = "+";
      commutable = TRUE;
      break;
    case OP_SUB:
      o = "-";
      break;
    case OP_MUL:
      o = "*";
      commutable = TRUE;
      break;
    case OP_DIV:
      o = "/";
      break;
    case OP_XTOI:
    case OP_XTOX:
      o = "**";
      break;
    case OP_CAT:
      o = "//";
      break;
    case OP_LEQV:
      o = ".eqv.";
      commutable = TRUE;
      break;
    case OP_LNEQV:
      o = ".neqv.";
      commutable = TRUE;
      break;
    case OP_LOR:
      o = ".or.";
      commutable = TRUE;
      break;
    case OP_LAND:
    case OP_SCAND:
      o = ".and.";
      commutable = TRUE;
      break;
    case OP_EQ:
      o = ".eq.";
      break;
    case OP_GE:
      o = ".ge.";
      break;
    case OP_GT:
      o = ".gt.";
      break;
    case OP_LE:
      o = ".le.";
      break;
    case OP_LT:
      o = ".lt.";
      break;
    case OP_NE:
      o = ".ne.";
      break;
    default:
      o = "<bop>";
      break;
    }
    if (commutable && (precedence(lop) > precedence(rop)) && !ast_is_comment) {
      int tmp;
      tmp = lop;
      lop = rop;
      rop = tmp;
    }
    prec_op = precedence(ast);
    encl = prec_op > left_precedence(lop, prec_op);
    if (encl)
      put_char('(');
    print_ast(lop);
    if (encl)
      put_char(')');
    if (op_space)
      put_char(' ');
    put_l_to_u(o);
    if (op_space)
      put_char(' ');
    encl = prec_op >= right_precedence(rop, prec_op);
    if (encl)
      put_char('(');
    print_ast(rop);
    if (encl)
      put_char(')');
    break;
  case A_UNOP:
    lop = A_LOPG(ast);
    prec_lop = precedence(lop);
    encl = precedence(ast) >= prec_lop;
    switch (A_OPTYPEG(ast)) {
    case OP_ADD:
      if (!encl && prec_lop != PREC_TERM)
        o = "+ ";
      else
        o = "+";
      break;
    case OP_SUB:
      if (negative_constant(lop))
        encl = TRUE;
      if (!encl && prec_lop != PREC_TERM)
        o = "- ";
      else
        o = "-";
      break;
    case OP_LNOT:
      o = ".not. ";
      break;
    case OP_LOC:
      print_loc(lop);
      return;
    case OP_REF:
      put_l_to_u("%ref(");
      goto un_builtin;
    case OP_BYVAL:
      put_l_to_u("%byval(");
      goto un_builtin;
    case OP_VAL:
      if (ast == astb.ptr0) {
        put_string("pghpf_0(3)");
        return;
      }
      if (ast == astb.ptr0c) {
        put_string("pghpf_0c");
        return;
      }
      put_l_to_u("%val(");
    un_builtin:
      print_ast(lop);
      put_char(')');
      return;
    default:
      o = "<uop>";
      break;
    }
    put_l_to_u(o);
    if (encl)
      put_char('(');
    print_ast(lop);
    if (encl)
      put_char(')');
    break;
  case A_CMPLXC:
    put_char('(');
    print_ast((int)A_LOPG(ast));
    put_char(',');
    print_ast((int)A_ROPG(ast));
    put_char(')');
    break;
  case A_CONV:
    print_ast((int)A_LOPG(ast));
    break;
  case A_PAREN:
    put_char('(');
    print_ast((int)A_LOPG(ast));
    put_char(')');
    break;
  case A_MEM:
    lop = (int)A_PARENTG(ast);
    print_ast(lop);
    dtype = A_DTYPEG(lop);
    if (DTYG(dtype) == TY_DERIVED)
      put_char('%');
    else
      put_char('.');
    print_ast(A_MEMG(ast));
    break;
  case A_SUBSCR:
    asd = A_ASDG(ast);
    lop = A_LOPG(ast);
    linearize = pr_chk_arr(lop);
    if (ast_is_comment)
      linearize = 0;
    if (XBIT(70, 8))
      linearize = 0;
    put_char('(');
    save_op_space = op_space;
    op_space = FALSE;
    if (linearize == 1) {
      /* if the output is standard f77, need to linearize the
       * subscripts for subscripting an allocatable array.
       */
      int asym, dsym;
      int dtype;
      ADSC *ad;
      int ln, lw, up, stride;

      asym = memsym_of_ast(lop);
      dsym = DESCRG(asym);
      dtype = DTYPEG(dsym);
      if (DTY(dtype) != TY_ARRAY)
        dtype = DTYPEG(asym);
      ad = AD_DPTR(dtype);
      dtype = DDTG(dtype); /* element type */
      i = ASD_NDIM(asd) - 1;
      lw = AD_LWAST(ad, i);
      if (lw == 0)
        lw = astb.i1;
      ln = mk_binop(OP_SUB, (int)ASD_SUBS(asd, i), lw, astb.bnd.dtype);

      for (i = i - 1; i >= 0; i--) {
        lw = AD_LWAST(ad, i);
        if (lw == 0)
          lw = astb.bnd.one;
        up = AD_UPAST(ad, i);
        if (up == 0)
          up = astb.bnd.one;
        stride = mk_binop(OP_SUB, up, lw, astb.bnd.dtype);
        stride = mk_binop(OP_ADD, stride, astb.bnd.one, astb.bnd.dtype);
        ln = mk_binop(OP_MUL, ln, stride, astb.bnd.dtype);

        /*  + (j - bnd) --> + j - bnd */
        ln = mk_binop(OP_ADD, ln, (int)ASD_SUBS(asd, i), astb.bnd.dtype);
        ln = mk_binop(OP_SUB, ln, lw, astb.bnd.dtype);
      }
      if (NO_CHARPTR && DTY(dtype) == TY_CHAR) {
        /* same as if the f77 output is not allowed to have pointers */
        if (ln != astb.bnd.zero) {
          print_ast(ln);
          put_char('+');
        }
        if (PTROFFG(asym)) {
          int offset;
          offset = check_member(lop, mk_id(PTROFFG(asym)));
          print_ast(offset);
        } else if (MIDNUMG(asym)) {
          int offset;
          offset = check_member(lop, mk_id(MIDNUMG(asym)));
          print_ast(offset);
        } else {
          put_int(1);
        }
      } else if (NO_DERIVEDPTR && DTY(dtype) == TY_DERIVED) {
        /* same as if the f77 output is not allowed to have pointers */
        if (ln != astb.bnd.zero) {
          print_ast(ln);
          put_char('+');
        }
        if (PTROFFG(asym))
          put_string(SYMNAME(PTROFFG(asym)));
        else if (MIDNUMG(asym))
          put_string(SYMNAME(MIDNUMG(asym)));
        else
          put_int(1);
      } else if (!NO_PTR) {
        /* for f77 output with pointers, need to add '1' to offset
         * the effect of the target compiler of subtracting 1 from
         * the linearized subscript expression.
         */
        ln = mk_binop(OP_ADD, ln, astb.bnd.one, astb.bnd.dtype);
        print_ast(ln);
      } else {
        /* for f77 output without pointers, add in the 'pointer offset';
         * added at the end of the subscript expression since the
         * expression could be 0 or a unary negate.  Note that a 1 is
         * unnecessary since the 'pointer offset' added to the array
         * is the base address of the allocated array.  The subscript
         * expression is just an offset from the base address.
         */
        if (ln != astb.bnd.zero) {
          print_ast(ln);
          put_char('+');
        }
        if (PTROFFG(asym)) {
          int offset;
          offset = check_member(lop, mk_id(PTROFFG(asym)));
          print_ast(offset);
        } else if (MIDNUMG(asym)) {
          int offset;
          offset = check_member(lop, mk_id(MIDNUMG(asym)));
          print_ast(offset);
        } else {
          put_int(1);
        }
      }
      put_char(')');
      op_space = save_op_space;
      break;
    } else if (linearize) {
      /* POINTER or nonPOINTER object has static descriptor */
      int asym;
      int dtyp;
      int lw, off, offset, str, acc1;
      int nd;
      LOGICAL no_mult;

      asym = memsym_of_ast(lop);
      dtyp = DTYPEG(asym);
      nd = ASD_NDIM(asd);
      no_mult = FALSE;
      if (nd == 1 && !POINTERG(asym) &&
          (!XBIT(58, 0x22) || NEWARGG(asym) == 0) /* not a remapped dummy */
          && SCG(asym) != SC_DUMMY)
        no_mult = TRUE;
      off = 0;
      if (no_mult) {
        lw = ASD_SUBS(asd, 0);
        acc1 = astb.bnd.zero;
        if (XBIT(58, 0x22) && ADD_LWAST(dtyp, 0))
          acc1 = mk_binop(OP_SUB, ADD_LWAST(dtyp, 0), astb.bnd.one,
                          astb.bnd.dtype);
        lw = mk_binop(OP_SUB, lw, acc1, astb.bnd.dtype);
        if (lw != astb.bnd.zero) {
          print_ast(lw);
          off = 1;
        }
      } else {
        for (i = 0; i < nd; i++) {
          lw = ASD_SUBS(asd, i);
          acc1 = astb.bnd.zero;
          if (XBIT(58, 0x22) && !POINTERG(asym) && ADD_LWAST(dtyp, i)) {
            acc1 = mk_binop(OP_SUB, ADD_LWAST(dtyp, i), astb.bnd.one,
                            astb.bnd.dtype);
          }
          lw = mk_binop(OP_SUB, lw, acc1, astb.bnd.dtype);
          str = check_member(lop, get_local_multiplier(linearize, i));
          if (lw != astb.bnd.zero) {
            if (off)
              put_char('+');
            if (lw != astb.bnd.one) {
              prec_op = left_precedence(lw, PREC_MULT);
              if (prec_op < PREC_MULT)
                put_char('(');
              print_ast(lw);
              if (prec_op < PREC_MULT)
                put_char(')');
              put_char('*');
            }
            print_ast(str);
            off = 1;
          }
          if (F77OUTPUT && XBIT(58, 0x22) && !POINTERG(asym) && NEWARGG(asym)) {
            /* a remapped dummy array argument;
             * have to also add section offset */
            if (off)
              put_char('+');
            off = 1;
            str = check_member(lop, get_section_offset(linearize, i));
            print_ast(str);
          }
        }
      }
      if (off)
        put_char('+');
      offset = check_member(lop, get_xbase(linearize));
      print_ast(offset);
      if (!POINTERG(asym) && SCG(asym) == SC_DUMMY) {
        put_char(')');
        op_space = save_op_space;
        break;
      }

      if (NO_PTR || (NO_CHARPTR && DTYG(DTYPEG(asym)) == TY_CHAR) ||
          (NO_DERIVEDPTR && DTYG(DTYPEG(asym)) == TY_DERIVED)) {
        put_char('+');
        if (PTROFFG(asym)) {
          offset = check_member(lop, mk_id(PTROFFG(asym)));
        } else {
          assert(MIDNUMG(asym),
                 "astout:linearize subscripts, midnum & ptroff 0", asym, 3);
          offset = check_member(lop, mk_id(MIDNUMG(asym)));
        }
        print_ast(offset);
        put_string("-1");
      }

      put_char(')');
      op_space = save_op_space;
      break;
    }
    for (i = 0; i < (int)ASD_NDIM(asd) - 1; i++) {
      print_ast((int)ASD_SUBS(asd, i));
      put_char(',');
    }
    print_ast((int)ASD_SUBS(asd, ASD_NDIM(asd) - 1));
    put_char(')');
    op_space = save_op_space;
    break;
  case A_SUBSTR:
    print_ast((int)A_LOPG(ast));
    put_char('(');
    if (A_LEFTG(ast))
      print_ast((int)A_LEFTG(ast));
    put_char(':');
    if (A_RIGHTG(ast))
      print_ast((int)A_RIGHTG(ast));
    put_char(')');
    break;
  case A_TRIPLE:
    /* [lb]:[ub][:stride] */
    if (A_LBDG(ast))
      print_ast((int)A_LBDG(ast));
    put_char(':');
    if (A_UPBDG(ast))
      print_ast((int)A_UPBDG(ast));
    if (A_STRIDEG(ast)) {
      put_char(':');
      print_ast((int)A_STRIDEG(ast));
    }
    break;
  case A_INTR:
    optype = A_OPTYPEG(ast);
    if (ast_is_comment) {
      if (A_ISASSIGNLHSG(ast)) {
        assert(optype == I_ALLOCATED, "unexpected ISASSIGNLHS", ast, ERR_Fatal);
        put_call(ast, 0, "allocated_lhs", 0);
      } else if (A_ISASSIGNLHS2G(ast)) {
        assert(optype == I_ALLOCATED, "unexpected ISASSIGNLHS2", ast,
               ERR_Fatal);
        put_call(ast, 0, "allocated_lhs2", 0);
      } else {
        put_call(ast, 0, NULL, 0);
      }
      break;
    }
    if ((sym = EXTSYMG(intast_sym[optype]))) {
      put_call(ast, 0, SYMNAME(sym), 0);
      break;
    }
    switch (optype) {
    case I_INT:
      dtype = DDTG(A_DTYPEG(ast));
      put_call(ast, 0, NULL, 0);
      break;
    case I_NINT:
      save_dtype = A_DTYPEG(ast);
      dtype = DDTG(save_dtype);
      put_call(ast, 0, NULL, 0);
      break;
    case I_REAL:
      save_dtype = A_DTYPEG(ast);
      dtype = DDTG(save_dtype);
      put_call(ast, 0, NULL, 0);
      break;
    case I_AINT:
    case I_ANINT:
      save_dtype = A_DTYPEG(ast);
      dtype = DDTG(save_dtype);
      argt = A_ARGSG(ast);
      i = ARGT_ARG(argt, 0);
      put_call(ast, 0, NULL, 0);
      break;
    case I_SIZE:
      argt = A_ARGSG(ast);
      shape = A_SHAPEG(ARGT_ARG(argt, 0));
      cnt = SHD_NDIM(shape);
      put_string(mkRteRtnNm(RTE_size));
      put_char('(');
      put_int((INT)cnt);
      put_char(',');
      print_ast((int)ARGT_ARG(argt, 1));
      for (i = 0; i < cnt - 1; i++) {
        put_char(',');
        print_ast((int)SHD_LWB(shape, i));
        put_char(',');
        print_ast((int)SHD_UPB(shape, i));
        put_char(',');
        print_ast((int)SHD_STRIDE(shape, i));
      }
      put_char(',');
      print_ast((int)SHD_LWB(shape, i));
      put_char(',');
      if (SHD_UPB(shape, i))
        print_ast((int)SHD_UPB(shape, i));
      else
        print_ast(astb.ptr0);
      put_char(',');
      print_ast((int)SHD_STRIDE(shape, i));
      put_char(')');
      break;
    case I_LBOUND:
    case I_UBOUND:
      argt = A_ARGSG(ast);
      shape = A_SHAPEG(ARGT_ARG(argt, 0));
      cnt = SHD_NDIM(shape);
      if (optype == I_LBOUND)
        put_string(mkRteRtnNm(RTE_lb));
      else
        put_string(mkRteRtnNm(RTE_ub));
      put_char('(');
      put_int((INT)cnt);
      put_char(',');
      print_ast((int)ARGT_ARG(argt, 1));
      for (i = 0; i < cnt; i++) {
        put_char(',');
        print_ast((int)SHD_LWB(shape, i));
        put_char(',');
        if (SHD_UPB(shape, i))
          print_ast((int)SHD_UPB(shape, i));
        else
          print_ast(astb.ptr0);
      }
      put_char(')');
      break;
    case I_CMPLX:
      argt = A_ARGSG(ast);
      if (ARGT_ARG(argt, 2) != 0 && ARGT_ARG(argt, 1) == 0) {
        /* Kind arg, no second parameter, f90 output */
        put_string("cmplx");
        put_char('(');
        print_ast(ARGT_ARG(argt, 0));
        put_char(',');
        put_string("kind");
        put_char('=');
        print_ast(ARGT_ARG(argt, 2));
        put_char(')');
        break;
      }
      save_dtype = A_DTYPEG(ast);
      dtype = DDTG(save_dtype);
      put_call(ast, 0, NULL, 0);
      break;
    case I_DIMAG:
      /* since LOP may be aimag, force the name 'dimag' */
      put_call(ast, 0, "dimag", 0);
      break;
    case I_INDEX:
      if (A_ARGCNTG(ast) != 2) {
        rtlRtn = RTE_indexa;
        goto make_func_name;
      }
      put_call(ast, 0, NULL, 0);
      break;
    case I_CEILING:
    case I_MODULO:
    case I_FLOOR:
      i = PNMPTRG(A_SPTRG(A_LOPG(ast))); /* locates "-<name>" */
      put_call(ast, 0, stb.n_base + i + 1, 0);
      break;
    case I_ALLOCATED:
      rtlRtn = RTE_allocated;
      goto make_func_name;
    case I_PRESENT:
      put_call(ast, 0, NULL, 2);
      break;
    case I_ACHAR:
      rtlRtn = RTE_achara;
      goto make_func_name;
    case I_EXPONENT:
      argt = A_ARGSG(ast);
      if (DTY(DDTG(A_DTYPEG(ARGT_ARG(argt, 0)))) == TY_REAL)
        rtlRtn = RTE_expon;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ARGT_ARG(argt, 0)))) == TY_QUAD)
        rtlRtn = RTE_exponq;
#endif
      else
        rtlRtn = RTE_expond;
      goto make_func_name;
    case I_FRACTION:
      if (DTY(DDTG(A_DTYPEG(ast))) == TY_REAL)
        rtlRtn = RTE_frac;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ast))) == TY_QUAD)
        rtlRtn = RTE_fracq;
#endif
      else
        rtlRtn = RTE_fracd;
      goto make_func_name;
    case I_IACHAR:
      rtlRtn = RTE_iachara;
      goto make_func_name;
    case I_RRSPACING:
      if (DTY(DDTG(A_DTYPEG(ast))) == TY_REAL)
        rtlRtn = RTE_rrspacing;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ast))) == TY_QUAD)
        rtlRtn = RTE_rrspacingq;
#endif
      else
        rtlRtn = RTE_rrspacingd;
      goto make_func_name;
    case I_SPACING:
      if (DTY(DDTG(A_DTYPEG(ast))) == TY_REAL)
        rtlRtn = RTE_spacing;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ast))) == TY_QUAD)
        rtlRtn = RTE_spacingq;
#endif
      else
        rtlRtn = RTE_spacingd;
      goto make_func_name;
    case I_NEAREST:
      if (DTY(DDTG(A_DTYPEG(ast))) == TY_REAL)
        rtlRtn = RTE_nearest;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ast))) == TY_QUAD)
        rtlRtn = RTE_nearestq;
#endif
      else
        rtlRtn = RTE_nearestd;
      goto make_func_name;
    case I_SCALE:
      if (DTY(DDTG(A_DTYPEG(ast))) == TY_REAL)
        rtlRtn = RTE_scale;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ast))) == TY_QUAD)
        rtlRtn = RTE_scaleq;
#endif
      else
        rtlRtn = RTE_scaled;
      goto make_func_name;
    case I_SET_EXPONENT:
      if (DTY(DDTG(A_DTYPEG(ast))) == TY_REAL)
        rtlRtn = RTE_setexp;
#ifdef TARGET_SUPPORTS_QUADFP
      else if (DTY(DDTG(A_DTYPEG(ast))) == TY_QUAD)
        rtlRtn = RTE_setexpq;
#endif
      else
        rtlRtn = RTE_setexpd;
      goto make_func_name;
    case I_VERIFY:
      argt = A_ARGSG(ast);
      if (DTY(DDTG(A_DTYPEG(ARGT_ARG(argt, 0)))) == TY_CHAR)
        rtlRtn = RTE_verifya;
      else
        rtlRtn = RTE_nverify;
      goto make_func_name;
    case I_SCAN:
      argt = A_ARGSG(ast);
      if (DTY(DDTG(A_DTYPEG(ARGT_ARG(argt, 0)))) == TY_CHAR)
        rtlRtn = RTE_scana;
      else
        rtlRtn = RTE_nscan;
      goto make_func_name;
    case I_LEN_TRIM:
      argt = A_ARGSG(ast);
      if (DTY(DDTG(A_DTYPEG(ARGT_ARG(argt, 0)))) == TY_CHAR)
        rtlRtn = RTE_lentrima;
      else
        rtlRtn = RTE_nlentrim;
      goto make_func_name;
    case I_ILEN:
      rtlRtn = RTE_ilen;
      goto make_func_name;
#ifdef I_LEADZ
    case I_LEADZ:
      /* Leadz, popcnt, and poppar are hpf_library and cray
       * intrinsics.  If the target is a cray, the cray versions supersede
       * the hpf versions.
       */
      if (XBIT(49, 0x1040000)) {
        /* T3D/T3E or C90 Cray targets */
        put_call(ast, 0, NULL, 0);
        break;
      }
      rtlRtn = RTE_leadz;
      goto make_func_name;
#endif
#ifdef I_TRAILZ
    case I_TRAILZ:
      if (XBIT(49, 0x1040000)) {
        /* T3D/T3E or C90 Cray targets */
        put_call(ast, 0, NULL, 0);
        break;
      }
      rtlRtn = RTE_trailz;
      goto make_func_name;
#endif
#ifdef I_POPCNT
    case I_POPCNT:
      if (XBIT(49, 0x1040000)) {
        /* T3D/T3E or C90 Cray targets */
        put_call(ast, 0, NULL, 0);
        break;
      }
      rtlRtn = RTE_popcnt;
      goto make_func_name;
#endif
#ifdef I_POPPAR
    case I_POPPAR:
      if (XBIT(49, 0x1040000)) {
        /* T3D/T3E or C90 Cray targets */
        put_call(ast, 0, NULL, 0);
        break;
      }
      rtlRtn = RTE_poppar;
/*****  fall thru  *****/
#endif
    make_func_name:
      put_call(ast, 0, mkRteRtnNm(rtlRtn), 0);
      break;
    case I_RESHAPE:
      /* this only occurs if the output is F90 */
      argt = A_ARGSG(ast);
      put_string(mkRteRtnNm(RTE_reshape));
      put_char('(');
      print_ast((int)ARGT_ARG(argt, 0));
      put_char(',');
      print_ast((int)ARGT_ARG(argt, 1));
      if (ARGT_ARG(argt, 2)) {
        put_char(',');
        put_string("pad=");
        print_ast((int)ARGT_ARG(argt, 2));
      }
      if (ARGT_ARG(argt, 3)) {
        put_char(',');
        put_string("order=");
        print_ast((int)ARGT_ARG(argt, 3));
      }
      put_char(')');
      break;
    default:
      put_call(ast, 0, NULL, 0);
      break;
    }
    break;
  case A_ICALL:
    if (ast_is_comment) {
      put_call(ast, 1, NULL, 0);
      break;
    }
    switch (A_OPTYPEG(ast)) {
    case I_MVBITS:
      /* call mvbits(from, frompos, len, to, topos)
       * becomes
       * call RTE_mvbits(from, frompos, len, to, topos,
       *     szfrom, szfrompos, szlen, sztopos)
       */
      put_l_to_u("call ");
      put_string(mkRteRtnNm(RTE_mvbits));
      put_char('(');
      argt = A_ARGSG(ast);
      for (i = 0; i <= 4; i++) {
        print_ast((int)ARGT_ARG(argt, i));
        put_char(',');
      }
      lop = ARGT_ARG(argt, 0); /* size of from/to */
      put_int(size_of(DDTG(A_DTYPEG(lop))));
      put_char(',');

      lop = ARGT_ARG(argt, 1); /* size of frompos */
      put_int(size_of(DDTG(A_DTYPEG(lop))));
      put_char(',');

      lop = ARGT_ARG(argt, 2); /* size of len */
      put_int(size_of(DDTG(A_DTYPEG(lop))));
      put_char(',');

      lop = ARGT_ARG(argt, 4); /* size of topos */
      put_int(size_of(DDTG(A_DTYPEG(lop))));

      put_char(')');
      break;

    case I_NULLIFY:
      argt = A_ARGSG(ast);
      lop = ARGT_ARG(argt, 0);
      sym = find_pointer_variable(lop);
      gen_nullify(lop, sym, !NO_PTR && STYPEG(sym) == ST_MEMBER);
      break;

    case I_PTR2_ASSIGN:
      argt = A_ARGSG(ast);
      cnt = A_ARGCNTG(ast);
      lop = ARGT_ARG(argt, 0); /* pointer */
      if (A_TYPEG(lop) == A_SUBSCR)
        lop = A_LOPG(lop);
      sym = find_pointer_variable(lop);
      put_l_to_u("call ");
      if (DTYG(DTYPEG(sym)) != TY_CHAR)
        rtlRtn = cnt == 5 ? RTE_ptr_assign : RTE_ptr_assignx;
      else
        rtlRtn = cnt == 5 ? RTE_ptr_assign_chara : RTE_ptr_assign_charxa;
      put_string(mkRteRtnNm(rtlRtn));
      put_char('(');

      put_mem_string(lop, SYMNAME(sym));
      put_char(',');

      lop = ARGT_ARG(argt, 1);
      sym = find_pointer_variable(lop);
      put_mem_string(lop, SYMNAME(sym)); /* static descriptor */
      put_char(',');

      lop = ARGT_ARG(argt, 2); /* target */
      if (STYPEG(sym) != ST_VAR && A_TYPEG(lop) == A_SUBSCR && A_SHAPEG(lop))
        lop = A_LOPG(lop);
      print_ast(lop);
      put_char(',');

      rop = ARGT_ARG(argt, 3); /* target's descriptor */
      print_ast(rop);

      /* section flag and other datatype arguments */
      for (i = 4; i < cnt; ++i) {
        put_char(',');
        lop = ARGT_ARG(argt, i);
        print_ast(lop);
      }
      if (XBIT(70, 0x20)) {
        lop = ARGT_ARG(argt, 1); /* descriptor */
        sym = find_pointer_variable(lop);
        if (DESCARRAYG(sym) && STYPEG(sym) == ST_MEMBER) {
          int osym;
          osym = VARIANTG(sym);
          if (osym > NOSYM && STYPEG(osym) == ST_MEMBER) {
            put_char(',');
            print_ast_replaced(lop, sym, osym);
            osym = VARIANTG(osym);
            if (osym > NOSYM && STYPEG(osym) == ST_MEMBER) {
              put_char(',');
              print_ast_replaced(lop, sym, osym);
            }
          }
        }
      }
      put_char(')');
      break;

    case I_PTR_COPYIN:
      /* astout needs to generate the call to copy a pointer in since
       * printing the ast of the dummy base will result in a subscript
       * reference which includes its offset. The argument needs to be
       * passed 'as is' (naked base).
       */
      argt = A_ARGSG(ast);
      sym = A_SPTRG(ARGT_ARG(argt, 3)); /* pointer */
      if (DTYG(DTYPEG(sym)) != TY_CHAR)
        rtlRtn = RTE_ptr_ina;
      else
        rtlRtn = RTE_ptr_in_chara;
      put_l_to_u("call ");
      put_string(mkRteRtnNm(rtlRtn));
      put_char('(');
      /*
       * call pghpf_ptr_in(rank, kind, len, db, dd, ab, ad)
       *
       * example: call pghpf_ptr_in(1,27,4,p,p$sd,p$bs,p$s0)
       *
       * argt 0: ast of rank (A_CNST)
       * argt 1: ast of kind (A_CNST)
       * argt 2: ast of len  (A_CNST)
       * argt 3: ast of dummy base (A_ID) - naked base
       * argt 4: ast of dummy static descriptor (A_ID)
       * argt 5: ast of actual base (A_ID)
       * argt 6: ast of actual static_descriptor (A_ID)
       */
      i = 0;
      while (TRUE) {
        lop = ARGT_ARG(argt, i);
        if (i == 3)
          put_string(SYMNAME(sym));
        else
          print_ast(lop);
        i++;
        if (i >= 7)
          break;
        put_char(',');
      }
      if (XBIT(70, 0x20)) {
        if (MIDNUMG(sym)) {
          put_char(',');
          put_string(SYMNAME(MIDNUMG(sym)));
        }
        if (PTROFFG(sym)) {
          put_char(',');
          put_string(SYMNAME(PTROFFG(sym)));
        }
      }
      put_char(')');
      break;

    case I_PTR_COPYOUT:
      /* astout needs to generate the call to copy a pointer out since
       * printing the ast of the dummy base will result in a subscript
       * reference which includes its offset. The argument needs to be
       * passed 'as is' (naked base).
       */
      argt = A_ARGSG(ast);
      sym = A_SPTRG(ARGT_ARG(argt, 0)); /* pointer */
      put_l_to_u("call ");
      if (DTYG(DTYPEG(sym)) != TY_CHAR)
        rtlRtn = RTE_ptr_out;
      else
        rtlRtn = RTE_ptr_out_chara;
      put_string(mkRteRtnNm(rtlRtn));
      put_char('(');
      /*
       * call pghpf_ptr_out(ab, ad, db, dd)
       *
       * example: call pghpf_ptr_out(p$bs, p$s0, p, p$sd)
       *
       * argt 0: ast of actual base (A_ID) - naked base
       * argt 1: ast of actual static descriptor (A_ID)
       * argt 2: ast of dummy base (A_ID)
       * argt 3: ast of dummy static_descriptor (A_ID)
       */
      i = 0;
      while (TRUE) {
        lop = ARGT_ARG(argt, i);
        if (i == 0)
          put_string(SYMNAME(sym));
        else
          print_ast(lop);
        i++;
        if (i >= 4)
          break;
        put_char(',');
      }
      if (XBIT(70, 0x20)) {
        if (MIDNUMG(sym)) {
          put_char(',');
          put_string(SYMNAME(MIDNUMG(sym)));
        }
        if (PTROFFG(sym)) {
          put_char(',');
          put_string(SYMNAME(PTROFFG(sym)));
        }
      }
      put_char(')');
      break;
    case I_COPYIN:
      /* print naked id as 5th argument */
      argt = A_ARGSG(ast);
      cnt = A_ARGCNTG(ast);
      put_l_to_u("call ");
      put_string(mkRteRtnNm(RTE_qopy_in));
      put_char('(');
      nid = FALSE;
      if (XBIT(57, 0x80)) {
        int arg2, arg4;
        arg2 = ARGT_ARG(argt, 2);
        arg4 = ARGT_ARG(argt, 4);
        if (arg2 == arg4) {
          nid = TRUE;
        } else if (A_TYPEG(arg2) == A_SUBSCR && A_LOPG(arg2) == arg4) {
          nid = TRUE;
        }
      }
      for (i = 0; i < cnt; ++i) {
        if (i)
          put_char(',');
        lop = ARGT_ARG(argt, i);
        if (nid && (i == 2 || i == 4)) {
          print_naked_id(lop);
        } else {
          print_ast(lop);
        }
      }
      put_char(')');
      break;
    case I_COPYOUT:
      /* print naked id as 1st argument */
      argt = A_ARGSG(ast);
      cnt = A_ARGCNTG(ast);
      put_l_to_u("call ");
      put_string(mkRteRtnNm(RTE_copy_out));
      put_char('(');
      nid = FALSE;
      if (XBIT(57, 0x80)) {
        int arg0, arg1;
        arg0 = ARGT_ARG(argt, 0);
        arg1 = ARGT_ARG(argt, 1);
        if (arg0 == arg1) {
          nid = TRUE;
        } else if (A_TYPEG(arg1) == A_SUBSCR && A_LOPG(arg1) == arg0) {
          nid = TRUE;
        }
      }
      for (i = 0; i < cnt; ++i) {
        if (i)
          put_char(',');
        lop = ARGT_ARG(argt, i);
        if (nid && (i == 0 || i == 1)) {
          print_naked_id(lop);
        } else {
          print_ast(lop);
        }
      }
      put_char(')');
      break;

    default:
      put_call(ast, 1, NULL, 0);
      break;
    }
    break;
  case A_CALL:
    put_call(ast, 1, NULL, 1);
    break;
  case A_FUNC:
    put_call(ast, 0, NULL, 1);
    break;
  case A_ENTRY:
    put_l_to_u("entry ");
    print_header((int)A_SPTRG(ast));
    if (XBIT(49, 0x1000) && !ast_is_comment)
      pghpf_entry((int)A_SPTRG(ast));
    break;
  case A_ASN:
    print_ast((int)A_DESTG(ast));
    put_string(" = ");
    print_uncoerced_const((int)A_SRCG(ast));
    if (XBIT(49, 0x1000000) && !ast_is_comment) {
      int sptr = sym_of_ast(A_DESTG(ast));
      if (POINTERG(sptr) || TARGETG(sptr)) {
        /* ...for T3D/T3E targets, assignment through an F90-pointer
         * requires a SUPPRESS directive to suppress node compiler
         * optimizations. */
        strcpy(lbuff, "cdir$ suppress ");
        strcat(lbuff, SYMNAME(sptr));
        col = strlen(lbuff);
      }
    }
    break;
  case A_IF:
    put_l_to_u("if (");
    print_ast((int)A_IFEXPRG(ast));
    put_string(") ");
    print_ast((int)A_IFSTMTG(ast));
    break;
  case A_IFTHEN:
    put_l_to_u("if (");
    print_ast((int)A_IFEXPRG(ast));
    put_l_to_u(") then");
    push_indent();
    break;
  case A_ELSE:
    pop_indent();
    put_l_to_u(astb.atypes[atype]);
    push_indent();
    break;
  case A_ELSEIF:
    pop_indent();
    put_l_to_u("elseif (");
    print_ast((int)A_IFEXPRG(ast));
    put_l_to_u(") then");
    push_indent();
    break;
  case A_ENDIF:
  case A_ENDWHERE:
  case A_ENDFORALL:
    pop_indent();
    goto single_kwd;
  case A_AIF:
    put_l_to_u("if (");
    print_ast((int)A_IFEXPRG(ast));
    put_string(") ");
    print_ast((int)A_L1G(ast));
    put_char(',');
    print_ast((int)A_L2G(ast));
    put_char(',');
    print_ast((int)A_L3G(ast));
    break;
  case A_GOTO:
    put_l_to_u("goto ");
    print_ast((int)A_L1G(ast));
    break;
  case A_CGOTO:
    put_l_to_u("goto (");
    astli = A_LISTG(ast);
    while (TRUE) {
      print_ast((int)ASTLI_AST(astli));
      astli = ASTLI_NEXT(astli);
      if (astli == 0)
        break;
      put_char(',');
    }
    put_string(") ");
    print_ast((int)A_LOPG(ast));
    break;
  case A_AGOTO:
    put_l_to_u("goto ");
    print_ast((int)A_LOPG(ast));
    astli = A_LISTG(ast);
    if (astli) {
      put_string(" (");
      while (TRUE) {
        print_ast((int)ASTLI_AST(astli));
        astli = ASTLI_NEXT(astli);
        if (astli == 0)
          break;
        put_char(',');
      }
      put_char(')');
    }
    break;
  case A_ASNGOTO:
    lop = A_SRCG(ast);
    assert(A_TYPEG(lop) == A_LABEL, "print_ast, src A_ASNGOTO not label", lop,
           3);
    if ((i = FMTPTG(A_SPTRG(lop))) && !ast_is_comment) {
      print_ast((int)A_DESTG(ast));
      put_string(" = ");
      print_loc_of_sym(i);
    } else {
      put_l_to_u("assign ");
      print_ast((int)A_SRCG(ast));
      put_l_to_u(" to ");
      print_ast((int)A_DESTG(ast));
    }
    break;
  case A_DO:
    put_l_to_u("do ");
    if (A_DOLABG(ast)) {
      print_ast((int)A_DOLABG(ast));
      put_char(' ');
    }
    print_ast((int)A_DOVARG(ast));
    put_string(" = ");
    print_uncoerced_const((int)A_M1G(ast));
    put_string(", ");
    print_uncoerced_const((int)A_M2G(ast));
    if (A_M3G(ast) && A_M3G(ast) != astb.i1) {
      put_string(", ");
      print_uncoerced_const((int)A_M3G(ast));
    }
    push_indent(); /* BLOCKDO */
    break;
  case A_DOWHILE:
    put_l_to_u("do ");
    if (A_DOLABG(ast)) {
      print_ast((int)A_DOLABG(ast));
      put_char(' ');
    }
    put_l_to_u("while ");
    put_char('(');
    print_ast((int)A_IFEXPRG(ast));
    put_char(')');
    push_indent(); /* BLOCKDO */
    break;
  case A_ENDDO:
    pop_indent(); /* BLOCKDO */
    goto single_kwd;
  case A_CONTINUE:
    goto single_kwd;
  case A_END:
    if (ast_is_comment)
      goto single_kwd;
    if (gbl.rutype != RU_BDATA && XBIT(49, 0x1000)) {
      /* pghpf_function_exit() */
      put_l_to_u("call ");
      put_string(mkRteRtnNm(RTE_function_exit));
      put_string("()");
    }
    if (gbl.rutype == RU_PROG) {
      put_l_to_u("call ");
      put_string(mkRteRtnNm(RTE_exit));
      put_string("(0)");
    }
    if (gbl.internal == 1) {
      put_l_to_u("contains");
      break;
    }
    if (gbl.internal) {
      switch (gbl.rutype) {
      case RU_PROG:
        put_l_to_u("endprogram");
        break;
      case RU_SUBR:
        put_l_to_u("endsubroutine");
        break;
      case RU_FUNC:
        put_l_to_u("endfunction");
        break;
      default:
        put_l_to_u("end");
        break;
      }
      break;
    }
    goto single_kwd;
  case A_STOP:
    put_l_to_u("stop");
    goto stop_pause;
  case A_PAUSE:
    put_l_to_u("pause");
  stop_pause:
    if (A_LOPG(ast)) {
      put_char(' ');
      print_ast((int)A_LOPG(ast));
    }
    break;
  case A_RETURN:
    put_l_to_u("return");
    if (A_LOPG(ast)) {
      put_char(' ');
      print_ast((int)A_LOPG(ast));
    }
    break;
  case A_ALLOC:
    /* For standard f77 output, always generate calls to the
     * allocate/deallocate run-time routines.  Otherwise, watch
     * for allocating allocatable arrays from a MODULE or
     * POINTERs; deallocate isn't necessary for MODULE allocatable
     * arrays if the output is pgftn since pgftn allows
     * deallocation of a pointer-based array.  */
    if (!ast_is_comment) {
      object = A_SRCG(ast);
      if (A_TYPEG(object) == A_SUBSCR) {
        sym = find_pointer_variable(A_LOPG(object));
      } else {
        sym = find_pointer_variable(object);
      }
      if (!F90POINTERG(sym)) {
        if (A_TKNG(ast) == TK_ALLOCATE) {
          int array = 0;
          if (sym && DTY(DTYPEG(sym)) == TY_ARRAY) {
            array = 1;
          }
          if (F77OUTPUT || POINTERG(sym) ||
              (array && (MDALLOCG(sym) || PTROFFG(sym))) ||
              (!array && ADJLENG(sym))) {
            gen_allocate(object, (int)A_LOPG(ast));
            return;
          }
        } else {
          /* watch for deallocating a POINTER */
          if (STYPEG(sym) == ST_MEMBER) {
            gen_deallocate(object, (int)A_LOPG(ast), sym, !NO_PTR);
            return;
          }
          if (F77OUTPUT || (POINTERG(sym) || ADJLENG(sym))) {
            gen_deallocate(object, (int)A_LOPG(ast), sym, 0);
            return;
          }
        }
      }
    }
    put_u_to_l(tokname[A_TKNG(ast)]);
    put_char('(');
    print_ast((int)A_SRCG(ast));
    if (A_LOPG(ast)) {
      put_l_to_u(", stat=");
      print_ast((int)A_LOPG(ast));
    }
    if (A_DESTG(ast)) {
      put_l_to_u(", pinned=");
      print_ast((int)A_DESTG(ast));
    }
    if (A_M3G(ast)) {
      put_l_to_u(", errmsg=");
      print_ast((int)A_M3G(ast));
    }
    if (A_STARTG(ast)) {
      put_l_to_u(", source=");
      print_ast((int)A_STARTG(ast));
    }
    if (A_FIRSTALLOCG(ast))
      put_string(", firstalloc");
    if (A_DALLOCMEMG(ast))
      put_string(", dallocmem");
    if (A_DEVSRCG(ast)) {
      put_string(", devsrc=");
      print_ast(A_DEVSRCG(ast));
    }
    if (A_ALIGNG(ast)) {
      put_string(", align=");
      print_ast(A_ALIGNG(ast));
    }
    put_char(')');
    if (!ast_is_comment && A_TKNG(ast) == TK_DEALLOCATE) {
      int sptr, object = A_SRCG(ast);
      if (A_TYPEG(object) == A_ID) {
        sptr = A_SPTRG(object);
        if (MIDNUMG(sptr) && !CCSYMG(MIDNUMG(sptr))) {
          put_string(SYMNAME(MIDNUMG(sptr)));
          put_string(" = 0");
        }
      } else if (A_TYPEG(object) == A_MEM) {
        sptr = A_SPTRG(A_MEMG(object));
        if (MIDNUMG(sptr) && !CCSYMG(MIDNUMG(sptr))) {
          print_ast_replaced(object, sptr, MIDNUMG(sptr));
          put_string(" = 0");
        }
      }
    }
    break;
  case A_WHERE:
    put_l_to_u("where (");
    print_ast((int)A_IFEXPRG(ast));
    put_char(')');
    if (A_IFSTMTG(ast)) {
      print_ast((int)A_IFSTMTG(ast));
      break;
    }
    push_indent();
    break;
  case A_ELSEFORALL:
    pop_indent();
    put_l_to_u("elseforall");
    push_indent();
    break;
  case A_ELSEWHERE:
    pop_indent();
    put_l_to_u("elsewhere");
    push_indent();
    break;
  case A_FORALL:
    put_l_to_u("forall (");
    astli = A_LISTG(ast);
    while (TRUE) {
      put_string(SYMNAME(ASTLI_SPTR(astli)));
      put_char('=');
      print_ast((int)ASTLI_TRIPLE(astli));
      astli = ASTLI_NEXT(astli);
      if (astli == 0)
        break;
      put_string(", ");
    }
    if (A_IFEXPRG(ast)) {
      put_string(", ");
      print_ast((int)A_IFEXPRG(ast));
    }
    put_char(')');
    if (A_IFSTMTG(ast)) {
      put_char(' ');
      print_ast((int)A_IFSTMTG(ast));
      break;
    }
    push_indent();
    break;
  single_kwd:
    put_l_to_u(astb.atypes[atype]);
    break;
  case A_REDIM:
    if ((F77OUTPUT || PTROFFG(memsym_of_ast(A_SRCG(ast)))) && !ast_is_comment) {
      /* for standard f77 output, generate assign the values implied
       * by the explict shape to the array's bound temporaries.
       */
      gen_bnd_assn((int)A_SRCG(ast));
      return;
    }
    put_l_to_u("redimension ");
    print_ast((int)A_SRCG(ast));
    break;
  case A_COMMENT:
    save_comment = ast_is_comment;
    ast_is_comment = TRUE;
    lbuff[0] = '!';
    print_ast((int)A_LOPG(ast));
    ast_is_comment = save_comment;
    break;
  case A_COMSTR: {
    /*  raw output -- watch for newlines */
    char ch;

    o = COMSTR(ast);
    col = 0;
    while ((ch = *o++)) {
      if (ch == '\n') {
        col = 0;
      } else
        lbuff[col++] = ch;
    }
  } break;
  case A_REALIGN:
    put_string("realign ");
    print_ast((int)A_LOPG(ast));
    put_string(" with alndsc ");
    put_int((INT)A_DTYPEG(ast));
    break;
  case A_REDISTRIBUTE:
    put_string("redistribute ");
    print_ast((int)A_LOPG(ast));
    put_string(" with dstdsc ");
    put_int((INT)A_DTYPEG(ast));
    break;
  case A_HLOCALIZEBNDS:
    put_string("hlocalizebnds(");
    if (A_LOPG(ast))
      print_ast((int)A_LOPG(ast));
    put_char(',');
    if (A_ITRIPLEG(ast))
      print_ast((int)A_ITRIPLEG(ast));
    put_char(',');
    if (A_OTRIPLEG(ast))
      print_ast((int)A_OTRIPLEG(ast));
    put_char(',');
    if (A_DIMG(ast))
      print_ast((int)A_DIMG(ast));
    put_char(')');
    break;
  case A_HALLOBNDS:
    put_string("hallobnds(");
    if (A_LOPG(ast))
      print_ast((int)A_LOPG(ast));
    put_char(')');
    break;
  case A_HCYCLICLP:
    put_string("hcycliclp(");
    if (A_LOPG(ast))
      print_ast((int)A_LOPG(ast));
    put_char(',');
    if (A_ITRIPLEG(ast))
      print_ast((int)A_ITRIPLEG(ast));
    put_char(',');
    if (A_OTRIPLEG(ast))
      print_ast((int)A_OTRIPLEG(ast));
    put_char(',');
    if (A_OTRIPLE1G(ast))
      print_ast((int)A_OTRIPLE1G(ast));
    put_char(',');
    if (A_DIMG(ast))
      print_ast((int)A_DIMG(ast));
    put_char(')');
    break;
  case A_HOFFSET:
    sym = memsym_of_ast(A_LOPG(ast)); /* pointer-based object */
    if (NO_PTR || (NO_CHARPTR && DTYG(DTYPEG(sym)) == TY_CHAR) ||
        (NO_DERIVEDPTR && DTYG(DTYPEG(sym)) == TY_DERIVED)) {
      put_l_to_u("call ");
      put_string(mkRteRtnNm(RTE_ptr_offset));
      put_char('(');
      print_ast((int)A_DESTG(ast)); /* name of pointer or offset
                                     * variable */
      put_char(',');
      print_ast((int)A_ROPG(ast)); /* name of pointer variable */
      put_char(',');
      print_ast((int)A_LOPG(ast)); /* name of object */
      put_char(',');
      if (PTRVG(sym))
        i = DT_PTR;
      else
        i = DTYG(DTYPEG(sym));
      put_int((INT)ty_to_lib[i]); /* run-time 'kind' of object */
      put_char(')');
    }
    break;
  case A_HSECT:
    put_string("hsect(");
    if (A_LOPG(ast))
      print_ast((int)A_LOPG(ast));
    put_char(',');
    if (A_BVECTG(ast))
      print_ast((int)A_BVECTG(ast));
    put_char(')');
    break;
  case A_HCOPYSECT:
    put_string("hcopysect(");
    if (A_DESTG(ast))
      print_ast((int)A_DESTG(ast));
    put_char(',');
    if (A_SRCG(ast))
      print_ast((int)A_SRCG(ast));
    put_char(',');
    if (A_DDESCG(ast))
      print_ast((int)A_DDESCG(ast));
    put_char(',');
    if (A_SDESCG(ast))
      print_ast((int)A_SDESCG(ast));
    put_char(')');
    break;
  case A_HPERMUTESECT:
    put_string("hpermutesect(");
    if (A_DESTG(ast))
      print_ast((int)A_DESTG(ast));
    put_char(',');
    if (A_SRCG(ast))
      print_ast((int)A_SRCG(ast));
    put_char(',');
    if (A_DDESCG(ast))
      print_ast((int)A_DDESCG(ast));
    put_char(',');
    if (A_SDESCG(ast))
      print_ast((int)A_SDESCG(ast));
    put_char(',');
    if (A_BVECTG(ast))
      print_ast((int)A_BVECTG(ast));
    put_char(')');
    break;
  case A_HOVLPSHIFT:
    put_string("hovlpshift(");
    if (A_SRCG(ast))
      print_ast((int)A_SRCG(ast));
    put_char(',');
    if (A_SDESCG(ast))
      print_ast((int)A_SDESCG(ast));
    put_char(')');
    break;
  case A_HGETSCLR:
    put_string("hgetsclr(");
    if (A_DESTG(ast))
      print_ast((int)A_DESTG(ast));
    put_char(',');
    if (A_SRCG(ast))
      print_ast((int)A_SRCG(ast));
    if (A_LOPG(ast)) {
      put_char(',');
      print_ast((int)A_LOPG(ast));
    }
    put_char(')');
    break;
  case A_HGATHER:
    put_string("hgather(");
    goto hscat;
  case A_HSCATTER:
    put_string("hscatter(");
  hscat:
    if (A_VSUBG(ast))
      print_ast((int)A_VSUBG(ast));
    put_char(',');
    if (A_DESTG(ast))
      print_ast((int)A_DESTG(ast));
    put_char(',');
    if (A_SRCG(ast))
      print_ast((int)A_SRCG(ast));
    put_char(',');
    if (A_DDESCG(ast))
      print_ast((int)A_DDESCG(ast));
    put_char(',');
    if (A_SDESCG(ast))
      print_ast((int)A_SDESCG(ast));
    put_char(',');
    if (A_MDESCG(ast))
      print_ast((int)A_MDESCG(ast));
    put_char(',');
    if (A_BVECTG(ast))
      print_ast((int)A_BVECTG(ast));
    put_char(')');
    break;
  case A_HCSTART:
    put_string("hcstart(");
    if (A_LOPG(ast))
      print_ast((int)A_LOPG(ast));
    put_char(',');
    if (A_DESTG(ast))
      print_ast((int)A_DESTG(ast));
    put_char(',');
    if (A_SRCG(ast))
      print_ast((int)A_SRCG(ast));
    put_char(')');
    break;
  case A_HCFINISH:
    put_string("hcfinish(");
    goto hcfree;
  case A_HCFREE:
    put_string("hcfree(");
  hcfree:
    if (A_LOPG(ast))
      print_ast((int)A_LOPG(ast));
    put_char(')');
    break;
  case A_HOWNERPROC:
    put_string("hownerproc(");
    print_ast(A_LOPG(ast));
    if (A_DIMG(ast)) {
      put_char(',');
      print_ast(A_DIMG(ast));
      put_char(',');
      print_ast(A_M1G(ast));
      put_char(',');
      print_ast(A_M2G(ast));
    }
    put_char(')');
    break;
  case A_HLOCALOFFSET:
    put_string("hlocaloffset(");
    print_ast(A_LOPG(ast));
    put_char(')');
    break;
  case A_MASTER:
    lbuff[0] = '!';
    put_string("master");
    break;
  case A_ENDMASTER:
    lbuff[0] = '!';
    cnt = A_ARGCNTG(ast);
    put_string("end master");
    if (cnt) {
      save_comment = ast_is_comment;
      ast_is_comment = TRUE;
      put_string(", copy(");
      argt = A_ARGSG(ast);
      for (i = 0; i < cnt; ++i) {
        if (i)
          put_char(',');
        lop = ARGT_ARG(argt, i);
        print_ast(lop);
      }
      put_char(')');
      ast_is_comment = save_comment;
    }
    break;
  case A_CRITICAL:
    lbuff[0] = '!';
    put_string("critical");
    break;
  case A_ENDCRITICAL:
    lbuff[0] = '!';
    put_string("end critical");
    break;
  case A_ATOMIC:
    lbuff[0] = '!';
    put_string("atomic update ");
    goto ast_atomic_common;
  case A_ATOMICCAPTURE:
    lbuff[0] = '!';
    put_string("atomic capture ");
    goto ast_atomic_common;
  case A_ATOMICREAD:
    lbuff[0] = '!';
    put_string("atomic read ");
    goto ast_atomic_common;
  case A_ATOMICWRITE:
    lbuff[0] = '!';
    put_string("atomic write ");
  ast_atomic_common:
    if (A_LOPG(ast)) {
      save_comment = ast_is_comment;
      ast_is_comment = TRUE;
      print_ast(A_LOPG(ast));
      ast_is_comment = save_comment;
    }
    break;
  case A_ENDATOMIC:
    lbuff[0] = '!';
    put_string("end atomic ");
    break;
  case A_MP_ATOMIC:
  case A_MP_ENDATOMIC:
    break;
  case A_MP_ATOMICREAD:
    lbuff[0] = '!';
    if (A_SRCG(ast)) {
      put_string(" src:");
      print_ast(A_SRCG(ast));
    }
    break;
  case A_MP_ATOMICWRITE:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_LOPG(ast)) {
      put_char(',');
      put_string(" lop:");
      print_ast(A_LOPG(ast));
    }
    put_char(',');
    if (A_ROPG(ast)) {
      put_string(" rop:");
      print_ast(A_ROPG(ast));
    }
    if (A_MEM_ORDERG(ast)) {
      put_string(" mem_order(");
      print_ast(A_MEM_ORDERG(ast));
      put_string(")");
    }
    break;
  case A_MP_ATOMICUPDATE:
  case A_MP_ATOMICCAPTURE:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_LOPG(ast)) {
      put_string(" lop:");
      print_ast(A_LOPG(ast));
    }
    put_char(',');
    if (A_ROPG(ast)) {
      put_string(" rop:");
      print_ast(A_ROPG(ast));
    }
    put_char(',');
    if (A_MEM_ORDERG(ast)) {
      put_string(" mem_order(");
      print_ast(A_MEM_ORDERG(ast));
      put_string(")");
    }
    break;

  case A_BARRIER:
    put_l_to_u("call ");
    put_string(mkRteRtnNm(RTE_barrier));
    put_string("()");
    break;
  case A_NOBARRIER:
    lbuff[0] = '!';
    put_string("no barrier");
    break;
  case A_MP_PARALLEL:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_IFPARG(ast)) {
      put_string(" if(");
      print_ast(A_IFPARG(ast));
      put_string(")");
    }
    if (A_NPARG(ast)) {
      put_string(" num_threads(");
      print_ast(A_NPARG(ast));
      put_string(")");
    }
    if (A_ENDLABG(ast)) {
      put_string(" endlab(");
      print_ast(A_ENDLABG(ast));
      put_string(")");
    }
    if (A_PROCBINDG(ast)) {
      put_string(" procbind(");
      print_ast(A_PROCBINDG(ast));
      put_string(")");
    }
    break;
  case A_MP_BMPSCOPE:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_STBLKG(ast)) {
      put_string(" st_block(");
      print_ast(A_STBLKG(ast));
      put_string(")");
    }
    break;
  case A_MP_TASK:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_IFPARG(ast)) {
      put_string(" if(");
      print_ast(A_IFPARG(ast));
      put_string(")");
    }
    if (A_FINALPARG(ast)) {
      put_string(" final(");
      print_ast(A_FINALPARG(ast));
      put_string(")");
    }
    if (A_PRIORITYG(ast)) {
      put_string(" priority(");
      print_ast(A_PRIORITYG(ast));
      put_string(")");
    }
    if (A_UNTIEDG(ast)) {
      put_string(",untied");
    }
    if (A_EXEIMMG(ast))
      put_string(",exeimm");
    if (A_ENDLABG(ast))
      print_ast(A_ENDLABG(ast));
    break;
  case A_MP_TASKLOOPREG:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_M1G(ast)) {
      put_string(" lb(");
      print_ast(A_M1G(ast));
      put_string(")");
    }
    if (A_M2G(ast)) {
      put_string(" ub(");
      print_ast(A_M2G(ast));
      put_string(")");
    }
    if (A_M3G(ast)) {
      put_string(" st(");
      print_ast(A_M3G(ast));
      put_string(")");
    }
    break;
  case A_MP_TASKLOOP:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_IFPARG(ast)) {
      put_string(" if(");
      print_ast(A_IFPARG(ast));
      put_string(")");
    }
    if (A_FINALPARG(ast)) {
      put_string(" final(");
      print_ast(A_FINALPARG(ast));
      put_string(")");
    }
    if (A_PRIORITYG(ast)) {
      put_string(" priority(");
      print_ast(A_PRIORITYG(ast));
      put_string(")");
    }
    if (A_UNTIEDG(ast)) {
      put_string(",untied");
    }
    if (A_NOGROUPG(ast)) {
      put_string(",nogroup");
    }
    if (A_GRAINSIZEG(ast)) {
      put_string(",grainsize");
    }
    if (A_NUM_TASKSG(ast)) {
      put_string(",num_tasks");
    }
    if (A_EXEIMMG(ast))
      put_string(",exeimm");
    if (A_ENDLABG(ast))
      print_ast(A_ENDLABG(ast));
    break;
  case A_MP_TASKFIRSTPRIV:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_LOPG(ast)) {
      put_string(" lop(");
      print_ast(A_LOPG(ast));
      put_string(")");
    }
    if (A_ROPG(ast)) {
      put_string(" rop(");
      print_ast(A_ROPG(ast));
      put_string(")");
    }
    break;

  case A_MP_TARGET:
  case A_MP_TARGETDATA:
  case A_MP_TARGETEXITDATA:
  case A_MP_TARGETENTERDATA:
  case A_MP_TARGETUPDATE:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_IFPARG(ast)) {
      put_string(" if(");
      print_ast(A_IFPARG(ast));
      put_string(")");
    }
    break;

  case A_MP_CANCEL:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_IFPARG(ast)) {
      put_string(" if(");
      print_ast(A_IFPARG(ast));
      put_string(")");
    }
    if (A_ENDLABG(ast)) {
      put_string(" endlab(");
      print_ast(A_ENDLABG(ast));
      put_string(")");
    }
    break;
  case A_MP_SECTIONS:
  case A_MP_CANCELLATIONPOINT:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_ENDLABG(ast)) {
      put_string(" endlab(");
      print_ast(A_ENDLABG(ast));
      put_string(")");
    }
    break;
  case A_MP_TASKREG:
  case A_MP_TASKDUP:
  case A_MP_ENDTARGET:
  case A_MP_ENDTARGETDATA:
  case A_MP_TEAMS:
  case A_MP_ENDTEAMS:
  case A_MP_DISTRIBUTE:
  case A_MP_ENDDISTRIBUTE:
  case A_MP_TASKGROUP:
  case A_MP_ETASKGROUP:
  case A_MP_ENDPARALLEL:
  case A_MP_BARRIER:
  case A_MP_ETASKDUP:
  case A_MP_TASKWAIT:
  case A_MP_TASKYIELD:
  case A_MP_ENDSECTIONS:
  case A_MP_SECTION:
  case A_MP_LSECTION:
  case A_MP_SINGLE:
  case A_MP_ENDSINGLE:
  case A_MP_MASTER:
  case A_MP_ENDMASTER:
  case A_MP_BCOPYIN:
  case A_MP_ECOPYIN:
  case A_MP_BCOPYPRIVATE:
  case A_MP_WORKSHARE:
  case A_MP_ENDWORKSHARE:
  case A_MP_BPDO:
  case A_MP_EPDO:
  case A_MP_BORDERED:
  case A_MP_EORDERED:
  case A_MP_ENDTASK:
  case A_MP_ETASKLOOP:
  case A_MP_EMPSCOPE:
  case A_MP_FLUSH:
  case A_MP_ETASKLOOPREG:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    break;
  case A_MP_TARGETLOOPTRIPCOUNT:
    put_string("target loop tripcount");
    break;
  case A_MP_MAP:
    put_string("map");
    break;
  case A_MP_EMAP:
    put_string("end map");
    break;
  case A_MP_BREDUCTION:
    put_string("begin reduction");
    break;
  case A_MP_EREDUCTION:
    put_string("end reduction");
    break;
  case A_MP_CRITICAL:
  case A_MP_ENDCRITICAL:
    lbuff[0] = '!';
    put_string(astb.atypes[atype]);
    if (A_MEMG(ast)) {
      put_char(' ');
      put_string(SYMNAME(A_MEMG(ast)));
    }
    break;
  case A_MP_PRE_TLS_COPY:
    lbuff[0] = '!';
    put_string("pre_tls_copy ");
    sym = A_SPTRG(ast);
    if (STYPEG(sym) == ST_CMBLK) {
      put_string("/");
      print_sname(sym);
      put_string("/");
    } else
      put_string(SYMNAME(sym));
    put_string(",size=");
    print_ast(A_ROPG(ast));
    break;
  case A_MP_COPYIN:
    lbuff[0] = '!';
    put_string("copyin ");
    sym = A_SPTRG(ast);
    if (STYPEG(sym) == ST_CMBLK) {
      put_string("/");
      print_sname(sym);
      put_string("/");
    } else
      put_string(SYMNAME(sym));
    put_string(",size=");
    print_ast(A_ROPG(ast));
    break;
  case A_MP_COPYPRIVATE:
    lbuff[0] = '!';
    put_string("copyprivate ");
    sym = A_SPTRG(ast);
    if (STYPEG(sym) == ST_CMBLK) {
      put_string("/");
      print_sname(sym);
      put_string("/");
    } else
      put_string(SYMNAME(sym));
    put_string(",size=");
    print_ast(A_ROPG(ast));
    break;
  case A_MP_PDO:
    lbuff[0] = '!';
    put_string("pdo");
    put_string(",sched=");
    put_intkind(A_SCHED_TYPEG(ast), DT_INT4);
    if (A_CHUNKG(ast)) {
      put_string(",chunk=");
      print_ast(A_CHUNKG(ast));
    }
    if (A_ORDEREDG(ast)) {
      put_string(",ordered");
    }
    if (A_ENDLABG(ast)) {
      print_ast(A_ENDLABG(ast));
    }
    A_TYPEP(ast, A_DO);
    put_string(" ");
    print_ast(ast);
    A_TYPEP(ast, A_MP_PDO);
    break;
  case A_MP_ENDPDO:
    pop_indent(); /* BLOCKDO */
    lbuff[0] = '!';
    put_string("endpdo");
    break;
  case A_PREFETCH:
    lbuff[0] = '!';
    put_string("prefetch ");
    print_ast(A_LOPG(ast));
    break;
  case A_PRAGMA:
    lbuff[0] = '!';
    lbuff[1] = 'p';
    lbuff[2] = 'g';
    lbuff[3] = 'i';
    lbuff[4] = '$';
    switch (A_PRAGMASCOPEG(ast)) {
    case PR_NOSCOPE:
      lbuff[5] = ' ';
      break;
    case PR_GLOBAL:
      lbuff[5] = 'g';
      break;
    case PR_ROUTINE:
      lbuff[5] = 'r';
      break;
    case PR_LOOP:
      lbuff[5] = 'l';
      break;
    case PR_LINE:
      lbuff[5] = 'n';
      break;
    }
    switch (A_PRAGMATYPEG(ast)) {
    case PR_NONE:
      print_ast(A_LOPG(ast));
      break;
    case PR_INLININGON:
      put_string("inline on");
      break;
    case PR_INLININGOFF:
      put_string("inline off");
      break;
    case PR_ALWAYSINLINE:
      put_string("inline always");
      break;
    case PR_NEVERINLINE:
      put_string("inline never");
      break;
    case PR_ACCBEGINDIR:
      acc_pragma(ast);
      put_string("begindir");
      break;
    case PR_ACCIMPDATAREG:
      acc_pragma(ast);
      put_string("implicit data region");
      break;
    case PR_ACCIMPDATAREGX:
      acc_pragma(ast);
      put_string("implicit data region(necessary)");
      break;
    case PR_ACCDATAREG:
      acc_pragma(ast);
      put_string("data");
      break;
    case PR_ACCHOSTDATA:
      acc_pragma(ast);
      put_string("host_data");
      break;
    case PR_ACCSCALARREG:
      acc_pragma(ast);
      put_string("scalar region");
      break;
    case PR_ACCSERIAL:
      acc_pragma(ast);
      put_string("serial");
      break;
    case PR_ACCENDSERIAL:
      acc_pragma(ast);
      put_string("end serial");
      break;
    case PR_ACCEL:
      acc_pragma(ast);
      put_string("region");
      break;
    case PR_ENDACCEL:
      acc_pragma(ast);
      put_string("end region");
      break;
    case PR_ACCENTERDATA:
      acc_pragma(ast);
      put_string("enter data");
      break;
    case PR_ACCEXITDATA:
      acc_pragma(ast);
      put_string("exit data");
      break;
    case PR_ACCFINALEXITDATA:
      acc_pragma(ast);
      put_string("exit data finalize");
      break;
    case PR_ACCENDDATAREG:
      acc_pragma(ast);
      put_string("end data");
      break;
    case PR_ACCENDHOSTDATA:
      acc_pragma(ast);
      put_string("end host_data");
      break;
    case PR_ACCENDSCALARREG:
      acc_pragma(ast);
      put_string("end scalar region");
      break;
    case PR_ACCENDIMPDATAREG:
      acc_pragma(ast);
      put_string("end implicit data region(");
      put_int(A_PRAGMAVALG(ast));
      put_string(")");
      break;
    case PR_INLINEONLY:
      put_string("inline only");
      break;
    case PR_INLINETYPE:
      put_string("inline type");
      break;
    case PR_INLINEAS:
      put_string("inline as");
      break;
    case PR_INLINEALIGN:
      put_string("inline align");
      break;
    case PR_ACCUPDATE:
      acc_pragma(ast);
      put_string("update");
      break;
    case PR_PCASTCOMPARE:
      acc_pragma(ast);
      put_string("comp");
      break;
    case PR_ACCWAIT:
      acc_pragma(ast);
      put_string("wait");
      break;
    case PR_ACCNOWAIT:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("nowait");
      break;
    case PR_ACCKERNELS:
      acc_pragma(ast);
      put_string("kernels");
      break;
    case PR_ACCENDKERNELS:
      acc_pragma(ast);
      put_string("end kernels");
      break;
    case PR_ACCPARCONSTRUCT:
      acc_pragma(ast);
      put_string("parallel");
      break;
    case PR_ACCENDPARCONSTRUCT:
      acc_pragma(ast);
      put_string("end parallel");
      break;
    case PR_ACCINDEPENDENT:
      acc_pragma(ast);
      put_string("independent");
      break;
    case PR_ACCAUTO:
      acc_pragma(ast);
      put_string("auto");
      break;
    case PR_ACCREDUCTOP:
      acc_pragma(ast);
      put_string("reduction operator(");
      switch (A_PRAGMAVALG(ast)) {
      case PR_ACCREDUCT_OP_ADD:
        put_string("+");
        break;
      case PR_ACCREDUCT_OP_MUL:
        put_string("*");
        break;
      case PR_ACCREDUCT_OP_MAX:
        put_string("max");
        break;
      case PR_ACCREDUCT_OP_MIN:
        put_string("min");
        break;
      case PR_ACCREDUCT_OP_BITAND:
        put_string("iand");
        break;
      case PR_ACCREDUCT_OP_BITIOR:
        put_string("ior");
        break;
      case PR_ACCREDUCT_OP_BITEOR:
        put_string("ieor");
        break;
      case PR_ACCREDUCT_OP_LOGAND:
        put_string(".and.");
        break;
      case PR_ACCREDUCT_OP_LOGOR:
        put_string(".or.");
        break;
      case PR_ACCREDUCT_OP_EQV:
        put_string(".eqv.");
        break;
      case PR_ACCREDUCT_OP_NEQV:
        put_string(".neqv");
        break;
      default:
        put_string("[unknown operator]");
        break;
      }
      put_string(")");
      break;
    case PR_ACCCOLLAPSE:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("collapse(");
      put_int(A_PRAGMAVALG(ast));
      put_string(")");
      break;
    case PR_ACCFORCECOLLAPSE:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("collapse(force:");
      put_int(A_PRAGMAVALG(ast));
      put_string(")");
      break;
    case PR_ACCTILE:
      acc_pragma(ast);
      acc_dtype(ast);
      cnt = A_ARGCNTG(ast);
      argt = A_ARGSG(ast);
      put_string("tile(");
      for (i = 0; i < cnt; ++i) {
        int arg;
        arg = ARGT_ARG(argt, i);
        if (i)
          put_string(",");
        print_ast(arg);
      }
      put_string(")");
      break;
    case PR_ACCPRIVATE:
    case PR_ACCFIRSTPRIVATE:
    case PR_ACCCOPY:
    case PR_ACCCOPYIN:
    case PR_ACCCOPYOUT:
    case PR_ACCLOCAL:
    case PR_ACCCREATE:
    case PR_ACCNO_CREATE:
    case PR_ACCPRESENT:
    case PR_ACCPCOPY:
    case PR_ACCPCOPYIN:
    case PR_ACCPCOPYOUT:
    case PR_ACCPCREATE:
    case PR_ACCPDELETE:
    case PR_ACCDELETE:
    case PR_ACCDEVICEPTR:
    case PR_ACCATTACH:
    case PR_ACCDETACH:
    case PR_ACCMIRROR:
    case PR_ACCREFLECT:
    case PR_ACCUPDATEHOST:
    case PR_ACCUPDATEHOSTIFP:
    case PR_ACCUPDATESELF:
    case PR_ACCUPDATESELFIFP:
    case PR_ACCUPDATEDEVICE:
    case PR_ACCUPDATEDEVICEIFP:
    case PR_ACCCOMPARE:
    case PR_PGICOMPARE:
    case PR_KERNEL_NEST:
    case PR_KERNEL_GRID:
    case PR_KERNEL_BLOCK:
    case PR_KERNEL_STREAM:
    case PR_KERNEL_DEVICE:
    case PR_ACCASYNC:
    case PR_ACCREDUCTION:
    case PR_ACCNUMWORKERS:
    case PR_ACCNUMGANGS:
    case PR_ACCNUMGANGS2:
    case PR_ACCNUMGANGS3:
    case PR_ACCVLENGTH:
    case PR_ACCUSEDEVICE:
    case PR_ACCUSEDEVICEIFP:
    case PR_ACCDEVICERES:
    case PR_ACCLOOPPRIVATE:
    case PR_CUFLOOPPRIVATE:
      acc_pragma(ast);
      switch (A_PRAGMATYPEG(ast)) {
      case PR_ACCPRIVATE:
        put_string("private(");
        break;
      case PR_ACCFIRSTPRIVATE:
        put_string("firstprivate(");
        break;
      case PR_ACCCOPY:
        put_string("copy(");
        break;
      case PR_ACCCOPYIN:
        put_string("copyin(");
        break;
      case PR_ACCCOPYOUT:
        put_string("copyout(");
        break;
      case PR_ACCLOCAL:
        put_string("local(");
        break;
      case PR_ACCCREATE:
        put_string("create(");
        break;
      case PR_ACCNO_CREATE:
        put_string("no_create(");
        break;
      case PR_ACCDELETE:
        put_string("delete(");
        break;
      case PR_ACCPRESENT:
        put_string("present(");
        break;
      case PR_ACCPCOPY:
        put_string("pcopy(");
        break;
      case PR_ACCPCOPYIN:
        put_string("pcopyin(");
        break;
      case PR_ACCPCOPYOUT:
        put_string("pcopyout(");
        break;
      case PR_ACCPCREATE:
        put_string("pcreate(");
        break;
      case PR_ACCPDELETE:
        put_string("pdelete(");
        break;
      case PR_ACCDEVICEPTR:
        put_string("deviceptr(");
        break;
      case PR_ACCATTACH:
        put_string("attach(");
        break;
      case PR_ACCDETACH:
        put_string("detach(");
        break;
      case PR_ACCUPDATEHOST:
        put_string("update host(");
        break;
      case PR_ACCUPDATEHOSTIFP:
        put_string("update if_present host(");
        break;
      case PR_ACCUPDATESELF:
        put_string("update self(");
        break;
      case PR_ACCUPDATESELFIFP:
        put_string("update if_present self(");
        break;
      case PR_ACCUPDATEDEVICE:
        put_string("update device(");
        break;
      case PR_ACCUPDATEDEVICEIFP:
        put_string("update if_present device(");
        break;
      case PR_ACCCOMPARE:
        put_string("acc_compare(");
        break;
      case PR_PGICOMPARE:
        put_string("pgi_compare(");
        break;
      case PR_ACCMIRROR:
        put_string("mirror(");
        break;
      case PR_ACCREFLECT:
        put_string("reflect(");
        break;
      case PR_KERNEL_NEST:
        cuf_pragma(ast);
        put_string("donest(");
        break;
      case PR_KERNEL_GRID:
        cuf_pragma(ast);
        put_string("grid(");
        break;
      case PR_KERNEL_BLOCK:
        cuf_pragma(ast);
        put_string("block(");
        break;
      case PR_KERNEL_STREAM:
        cuf_pragma(ast);
        put_string("stream(");
        break;
      case PR_KERNEL_DEVICE:
        cuf_pragma(ast);
        put_string("device(");
        break;
      case PR_ACCASYNC:
        acc_dtype(ast);
        put_string("async(");
        break;
      case PR_ACCREDUCTION:
        put_string("reduction(");
        break;
      case PR_ACCNUMWORKERS:
        acc_dtype(ast);
        put_string("num_workers(");
        break;
      case PR_ACCNUMGANGS:
        acc_dtype(ast);
        put_string("num_gangs(");
        break;
      case PR_ACCNUMGANGS2:
        acc_dtype(ast);
        put_string("num_gangs(dim:2,");
        break;
      case PR_ACCNUMGANGS3:
        acc_dtype(ast);
        put_string("num_gangs(dim:3,");
        break;
      case PR_ACCVLENGTH:
        acc_dtype(ast);
        put_string("vector_length(");
        break;
      case PR_ACCUSEDEVICE:
      case PR_ACCUSEDEVICEIFP:
        put_string("use_device(");
        break;
      case PR_ACCDEVICERES:
        put_string("device_resident(");
        break;
      case PR_ACCLINK:
        put_string("link(");
        break;
      case PR_ACCLOOPPRIVATE:
        put_string("loopprivate(");
        break;
      case PR_CUFLOOPPRIVATE:
        cuf_pragma(ast);
        put_string("loopprivate(");
        break;
      }
      print_ast(A_LOPG(ast));
      if (A_ROPG(ast)) {
        put_string(",");
        print_ast(A_ROPG(ast));
      }
      put_string(")");
      switch (A_PRAGMATYPEG(ast)) {
      case PR_ACCUSEDEVICEIFP:
      case PR_ACCUPDATEHOSTIFP:
      case PR_ACCUPDATESELFIFP:
      case PR_ACCUPDATEDEVICEIFP:
        put_string(" if_present");
        break;
      default :
        break;
      }
      break;
    case PR_KERNELBEGIN:
      cuf_pragma(ast);
      put_string("begin");
      break;
    case PR_KERNELTILE:
      cuf_pragma(ast);
      put_string("tile");
      break;
    case PR_ACCVECTOR:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop vector");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCWORKER:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop worker");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCGANG:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop gang");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCGANGDIM:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop gang");
      if (A_LOPG(ast)) {
        put_string("(dim:");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCGANGCHUNK:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop gang");
      if (A_LOPG(ast)) {
        put_string("(static:");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCPARALLEL:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop parallel");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCSEQ:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop seq");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCHOST:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop host");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCIF:
      acc_pragma(ast);
      put_string("if");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCUNROLL:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop unroll");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCSEQUNROLL:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop sequnroll");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCPARUNROLL:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop parunroll");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCVECUNROLL:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("loop vecunroll");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_KERNEL:
      cuf_pragma(ast);
      put_string("kernel");
      break;
    case PR_ENDKERNEL:
      cuf_pragma(ast);
      put_string("end kernel");
      break;
    case PR_ACCELLP:
      acc_pragma(ast);
      put_string("loop");
      break;
    case PR_ACCKLOOP:
      acc_pragma(ast);
      put_string("(kernels) loop");
      break;
    case PR_ACCTKLOOP:
      acc_pragma(ast);
      put_string("(kernels-tight) loop");
      break;
    case PR_ACCPLOOP:
      acc_pragma(ast);
      put_string("(parallel) loop");
      break;
    case PR_ACCTPLOOP:
      acc_pragma(ast);
      put_string("(parallel-tight) loop");
      break;
    case PR_ACCSLOOP:
      acc_pragma(ast);
      put_string("(serial) loop");
      break;
    case PR_ACCTSLOOP:
      acc_pragma(ast);
      put_string("(serial-tight) loop");
      break;
    case PR_ACCWAITDIR:
      acc_pragma(ast);
      put_string("waitdir");
      break;
    case PR_ACCWAITARG:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("wait");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCDEVICEID:
      acc_pragma(ast);
      acc_dtype(ast);
      put_string("deviceid");
      if (A_LOPG(ast)) {
        put_string("(");
        print_ast(A_LOPG(ast));
        put_string(")");
      }
      break;
    case PR_ACCCACHEDIR:
      acc_pragma(ast);
      put_string("cachedir");
      break;
    case PR_ACCCACHEREADONLY:
      acc_pragma(ast);
      put_string("cache-readonly");
      break;
    case PR_ACCCACHEARG:
      acc_pragma(ast);
      put_string("cache(");
      print_ast(A_LOPG(ast));
      put_string(")");
      break;
    case PR_ACCDEFNONE:
      acc_pragma(ast);
      put_string("default(none)");
      break;
    case PR_ACCDEFPRESENT:
      acc_pragma(ast);
      put_string("default(present)");
      break;
    default:
      put_string("pragmatype=");
      put_int(A_PRAGMATYPEG(ast));
      break;
    }
    break;
  default:
    put_string("ASTTYPE(");
    put_int(atype);
    put_string(")");
  }
}

static void
put_call(int ast, int call, const char *name, int check_ptrarg)
{
  int dpdsc, paramct, iface;
  int sptr, cnt, argt, arg, i, param, sdparam, sdi;
  LOGICAL anyoptional, do_naked_pointer, some;
  if (call) {
    put_l_to_u("call ");
  }
  if (name) {
    put_string(name);
  } else {
    print_ast(A_LOPG(ast));
  }
  put_char('(');
  sptr = procsym_of_ast(A_LOPG(ast));
  proc_arginfo(sptr, &paramct, &dpdsc, &iface);
  cnt = A_ARGCNTG(ast);
  argt = A_ARGSG(ast);
  altret_spec = TRUE;
  anyoptional = FALSE;
  sdi = -1; /* section descriptor index */
  /* f77 output, no pointers allowed, subprogram has a pointer argument */
  if (check_ptrarg == 1 && F77OUTPUT && NO_PTR && !ast_is_comment &&
      PTRARGG(sptr) && dpdsc > 0 && paramct > 0) {
    do_naked_pointer = TRUE;
  } else {
    do_naked_pointer = FALSE;
  }
  arg = 0;
  some = FALSE;
  for (i = 0; i < cnt; ++i) {
    /* if there was a previous argument, put comma */
    arg = ARGT_ARG(argt, i);
    if (i >= paramct && dpdsc) {
      ++sdi;
      {
        /* move sdi up to next assumed-shape argument */
        while (sdi < paramct) {
          sdparam = aux.dpdsc_base[dpdsc + sdi];
          if (sdparam && DTY(DTYPEG(sdparam)) == TY_ARRAY &&
              ASSUMSHPG(sdparam)) {
            break;
          } else {
            ++sdi;
          }
        }
      }
    }
    /* separate all arguments with comma */
    if (ast_is_comment && i)
      put_char(',');
    if (arg != 0) {
      param = 0;
      /* is this a missing optional argument? */
      if (i < paramct && dpdsc) {
        param = aux.dpdsc_base[dpdsc + i];
      }
      if (param && OPTARGG(param) && (arg == astb.ptr0 || arg == astb.ptr0c)) {
        /* don't print the missing argument */
        anyoptional = TRUE;
        arg = 0; /* don't print next comma */
      } else {
        /* separate all arguments with comma, unless already printed above */
        if (some && !ast_is_comment)
          put_char(',');
        some = TRUE;
        if (anyoptional) { /* must use keyword form */
          if (param) {
            put_string(SYMNAME(param));
            put_string("=");
          } else if (sdi >= 0 && sdi < paramct) {
            static char sdname[120];
            int sdparam;
            sdparam = aux.dpdsc_base[dpdsc + sdi];
            strcpy(sdname, SYMNAME(sdparam));
            strcat(sdname, "$sd");
            put_string(sdname);
            put_string("=");
          } else {
            put_string("NOKEYWORD=");
          }
        }
        if (check_ptrarg == 2 ||
            (do_naked_pointer && i < paramct && POINTERG(param))) {
          print_naked_id(arg);
        } else {
          print_ast(arg);
        }
      }
    }
  }
  altret_spec = FALSE;
  put_char(')');
} /* put_call */

static void
print_ast_replaced(int ast, int sym, int replacesym)
{
  int astreplace;
  if (replacesym && STYPEG(replacesym) != ST_MEMBER) {
    put_string(SYMNAME(replacesym));
  } else {
    /* replace 'sym' in 'ast' by 'replacesym', then print it */
    switch (A_TYPEG(ast)) {
    case A_ID:
    case A_CNST:
    case A_LABEL:
      astreplace = ast;
      break;
    case A_MEM:
      astreplace = A_MEMG(ast);
      if (A_TYPEG(astreplace) != A_ID)
        astreplace = 0;
      break;
    case A_SUBSCR:
      astreplace = A_LOPG(ast);
      if (A_TYPEG(astreplace) == A_MEM) {
        astreplace = A_MEMG(astreplace);
      }
      if (A_TYPEG(astreplace) != A_ID)
        astreplace = 0;
      break;
    default:
      astreplace = 0;
      break;
    }
    if (astreplace) {
      if (A_SPTRG(astreplace) == sym) {
        A_SPTRP(astreplace, replacesym);
      } else {
        astreplace = 0;
      }
    }
    print_ast(ast);
    if (astreplace) {
      A_SPTRP(astreplace, sym);
    }
  }
} /* print_ast_replaced */

static void
print_uncoerced_const(int ast)
{
  /*
   * Do not check the ALIAS field of the AST -- need to examine the actual
   * ast and not, for example, a convert ast which resolves to a constant.
   * Checking the ALIAS field of
   *     rrr = 4habcd
   * will result in emitting the 'real' representation of the Hollerith
   * constant, which is not desired.
   *
   */
  if (A_TYPEG(ast) == A_CNST) {
    put_const(A_SPTRG(ast));
    return;
  }
  print_ast(ast);
}

static void
print_loc(int ast)
{
  if (A_TYPEG(ast) == A_ID) {
    print_loc_of_sym(A_SPTRG(ast));
    return;
  }
  if (ast_is_comment) {
    put_string("loc");
  } else {
    put_string(mkRteRtnNm(RTE_loc));
  }
  put_char('(');
  print_ast(ast);
  put_char(')');
}

static void
print_loc_of_sym(int sym)
{
  if (SCG(sym) == SC_BASED && F77OUTPUT && !NO_PTR && MIDNUMG(sym) &&
      !ast_is_comment) {
    put_string(SYMNAME(MIDNUMG(sym)));
    return;
  }
  if (ast_is_comment) {
    put_string("loc");
  } else {
    put_string(mkRteRtnNm(RTE_loc));
  }
  put_char('(');
  print_refsym(sym, 0);
  put_char(')');
}

static void
print_refsym(int sym, int ast)
{
  if (F77OUTPUT && !ast_is_comment && !F90POINTERG(sym) &&
      (ALLOCG(sym) || SCG(sym) == SC_BASED ||        /* allocatable symbol */
       (STYPEG(sym) == ST_MEMBER && ALIGNG(sym)))) { /*dist member*/
    /* pgftn-extensions not allowed: cray pointers not allowed,
     * or cray pointers are allowed but the objects can't be character
     * or derived type.
     */
    if (NO_PTR || /* no pointers */
        (NO_CHARPTR && DTYG(DTYPEG(sym)) == TY_CHAR) ||
        (NO_DERIVEDPTR && DTYG(DTYPEG(sym)) == TY_DERIVED)) {
      put_string(SYMNAME(sym));
      put_char('(');
      if (PTROFFG(sym)) {
        int offset;
        offset = check_member(ast, mk_id(PTROFFG(sym)));
        print_ast(offset);
      } else {
        int offset;
        offset = check_member(ast, mk_id(MIDNUMG(sym)));
        print_ast(offset);
      }
      put_char(')');
      return;
    }
  }
  print_sname(sym);
  if (DBGBIT(5, 0x40)) {
    char b[64];
    sprintf(b, "\\%d", sym);
    put_string(b);
  }
}

static void
print_sname(int sym)
{
  switch (STYPEG(sym)) {
  case ST_MEMBER:
    break;
  case ST_PROC:
    if (SCOPEG(sym) && STYPEG(SCOPEG(sym)) == ST_ALIAS && SCOPEG(SCOPEG(sym)) &&
        STYPEG(SCOPEG(SCOPEG(sym))) == ST_MODULE) {
      put_string(SYMNAME(SCOPEG(SCOPEG(sym))));
      put_string("::");
      break;
    }
    FLANG_FALLTHROUGH;
  default:
    if (ENCLFUNCG(sym) && STYPEG(ENCLFUNCG(sym)) == ST_MODULE) {
      put_string(SYMNAME(ENCLFUNCG(sym)));
      put_string("::");
    }
    break;
  }
  switch (STYPEG(sym)) {
  case ST_UNKNOWN:
  case ST_IDENT:
  case ST_VAR:
  case ST_ARRAY:
  case ST_DESCRIPTOR:
  case ST_STRUCT:
  case ST_UNION:
    if (SCG(sym) == SC_PRIVATE)
      put_string("@");
    else if (SCG(sym) == SC_BASED && MIDNUMG(sym) &&
             SCG(MIDNUMG(sym)) == SC_PRIVATE)
      put_string("@");
    break;
  default:;
  }
  put_string(SYMNAME(sym));
}

static void
print_naked_id(int ast)
{
  if (A_TYPEG(ast) == A_ID) {
    int sym = A_SPTRG(ast);
    put_string(SYMNAME(sym));
  } else {
    print_ast(ast);
  }
}

/** \brief Since the output is 'standard' f77, all allocatable (deferred-shape)
    arrays must be converted to pointer-based arrays.  The symbol table
    is scanned to find allocatable arrays which do not have bound temporaries
    or associated pointer variables.
 */
void
deferred_to_pointer(void)
{
  int sptr;
  int dtype;
  int numdim;
  int i;
  ADSC *ad;

  for (sptr = stb.stg_avail - 1; sptr >= stb.firstusym; sptr--) {
    if (STYPEG(sptr) != ST_ARRAY || SCG(sptr) == SC_NONE)
      continue;
    if (IGNOREG(sptr)) /* ignore this symbol */
      continue;
    if (F90POINTERG(sptr))
      continue;
    dtype = DTYPEG(sptr);
    ad = AD_DPTR(dtype);
    if (!AD_DEFER(ad) && !AD_NOBOUNDS(ad))
      continue;

    numdim = AD_NUMDIM(ad);
    if (!ALIGNG(sptr) && SDSCG(sptr) == 0)
      /* if the array has a static descriptor, then never change the
       * bounds.
       */
      for (i = 0; i < numdim; ++i) {
        int s;
        if (AD_LWAST(ad, i) == 0 || A_TYPEG(AD_LWAST(ad, i)) != A_ID) {
          AD_LWAST(ad, i) = mk_bnd_ast();
          if (SAVEG(sptr)) {
            s = A_SPTRG(AD_LWAST(ad, i));
            SCP(s, SC_STATIC);
            SAVEP(s, 1);
          }
        }
        if (AD_UPAST(ad, i) == 0 || A_TYPEG(AD_UPAST(ad, i)) != A_ID) {
          AD_UPAST(ad, i) = mk_bnd_ast();
          if (SAVEG(sptr)) {
            s = A_SPTRG(AD_UPAST(ad, i));
            SCP(s, SC_STATIC);
            SAVEP(s, 1);
          }
        }
        if (AD_EXTNTAST(ad, i) == 0 || A_TYPEG(AD_EXTNTAST(ad, i)) != A_ID) {
          AD_EXTNTAST(ad, i) = mk_bnd_ast();
          if (SAVEG(sptr)) {
            s = A_SPTRG(AD_EXTNTAST(ad, i));
            SCP(s, SC_STATIC);
            SAVEP(s, 1);
          }
        }
      }
    /* don't create pointer variable for sequential dummy */
    /* or caller remapping dummys */
    if (SCG(sptr) == SC_DUMMY) {
      if (SEQG(sptr))
        continue;
      if (XBIT(58, 0x20) && !POINTERG(sptr))
        continue;
    }
    ALLOCP(sptr, 1);
    if (MIDNUMG(sptr) == 0) {
      int stp;
      SCP(sptr, SC_BASED);
      stp = sym_get_ptr(sptr);
      MIDNUMP(sptr, stp);
    }
    if (SAVEG(sptr)) {
      if (!POINTERG(sptr)) {
        SAVEP(MIDNUMG(sptr), 1);
      }
      if (!NO_PTR)
        /* pointers allowed in output! */
        SAVEP(sptr, 0); /* based-object cannot be SAVEd */
    }
  }
}

static void
pr_arr_name(int arr)
{
  int lop, sptr = 0, dtype;
  if (A_TYPEG(arr) == A_ID) {
    sptr = A_SPTRG(arr);
  } else if (A_TYPEG(arr) == A_MEM) {
    lop = A_PARENTG(arr);
    print_ast(lop);
    dtype = A_DTYPEG(lop);
    if (DTYG(dtype) == TY_DERIVED) {
      put_char('%');
    } else {
      put_char('.');
    }
    sptr = A_SPTRG(A_MEMG(arr));
  }
  print_sname(sptr);
} /* pr_arr_name */

/* a subscript ast is being processed.  First, print the array ('arr') which
 * is being subscripted and then check the array to determine if its subscripts
 * must be linearized.  Returns a non-zero value if the array's subscripts
 * must be linearized; 0, otherwise.  The non-zero value is 1 for non-POINTER
 * arrays; if the array is a POINTER, then the non-zero value is the sym
 * pointer representing the POINTER's static descriptor.
 */
static int
pr_chk_arr(int arr)
{
  int sptr = 0;
  if (A_TYPEG(arr) == A_ID) {
    sptr = A_SPTRG(arr);
  } else if (A_TYPEG(arr) == A_MEM) {
    sptr = A_SPTRG(A_MEMG(arr));
  } else {
    print_ast(arr);
    return 0;
  }
  if (LNRZDG(sptr)) {
    /* linearize flag set */
    pr_arr_name(arr);
    if (SDSCG(sptr) && !NODESCG(sptr))
      return SDSCG(sptr);
    return 1;
  } else if (F77OUTPUT) {
    if (ALLOCG(sptr) ||
        (SCG(sptr) == SC_BASED &&
         (NO_PTR || (NO_CHARPTR && DTYG(DTYPEG(sptr)) == TY_CHAR) ||
          (NO_DERIVEDPTR && DTYG(DTYPEG(sptr)) == TY_DERIVED)))) {
      pr_arr_name(arr);
      if (SDSCG(sptr) && !NODESCG(sptr))
        return SDSCG(sptr);
      return 1;
    }
  } else if (ALLOCG(sptr) && SCG(sptr) == SC_BASED &&
             (MDALLOCG(sptr) || PTROFFG(sptr))) {
    /* linearize subscripts of an allocatable array which came from
     * a MODULE.
     */
    pr_arr_name(arr);
    if (SDSCG(sptr) && !NODESCG(sptr))
      return SDSCG(sptr);
    return 1;
  }

  pr_arr_name(arr);
  return 0;
}

/* 'sub' is a subscript ast, where the array is allocatable. If the output
 * is standard f77, will need to generate assignment statements which assign
 * to the array's bound temporaries their respective values.  The values are
 * presented as 'triple' asts, representing the explicit shape of the array.
 * The bound temporaries are extracted from the LWAST and UPAST fields
 * of the array's descriptor (ADSC).
 */
static void
gen_bnd_assn(int sub)
{
  int i, ndim;
  int asd;
  int asym, dsym;
  ADSC *ad;
  int triple;
  int dtyp;

  if (A_TYPEG(sub) != A_SUBSCR) {
    return;
  }
  asd = A_ASDG(sub);
  ndim = ASD_NDIM(asd);
  asym = memsym_of_ast(A_LOPG(sub));
  dsym = DESCRG(asym);
  assert(dsym, "gen_bnd_assn: descr not found", asym, 4);
  dtyp = DDTG(DTYPEG(asym));
  dtyp = get_array_dtype(ndim, dtyp);
  DTYPEP(dsym, dtyp);
  ad = AD_DPTR(dtyp);
  assert(ndim == AD_NUMDIM(ad), "gen_bnd_assn:ndim not equal", asym, 3);
  for (i = 0; i < ndim; i++) {
    triple = ASD_SUBS(asd, i);
    if (A_TYPEG(triple) != A_TRIPLE) {
      return;
    }
    AD_LWAST(ad, i) = A_LBDG(triple);
    AD_UPAST(ad, i) = A_UPBDG(triple);
    AD_EXTNTAST(ad, i) = mk_extent(AD_LWAST(ad, i), AD_UPAST(ad, i), i);
  }
}

static int
find_member_base(int dtype)
{
  int basesptr, dty, mem;
  const char *rtnNm = mkRteRtnNm(RTE_member_base);
  basesptr = lookupsymbol(rtnNm);
  if (basesptr == 0 || STYPEG(basesptr) != ST_CMBLK) {
    return NOSYM;
  }
  /* find the member base */
  dty = DDTG(dtype);
  for (mem = CMEMFG(basesptr); mem > NOSYM; mem = SYMLKG(mem)) {
    if (DDTG(DTYPEG(mem)) == dty)
      break;
  }
  return mem;
} /* find_member_base */

/* If the output is 'standard' f77, need to convert the allocate of an
 * object to a call to a run-time routine.  'object' is the ast item
 * representing the object; 'stat' is the id ast, not present if 0, of the
 * allocate status variable.  If 'object' is a subscript ast, the subscripts
 * are triples (represents the explicit shape of the allocate); prior to
 * calling the run-time routine, the values specified by the explicit
 * shape must be assigned to the array's bound temporaries. Otherwise, 'object'
 * is an id ast, whose symbol field is the array to be allocate.
 * Note that the object is a pointer-based array; the associated pointer
 * variable is assigned the pointer of the allocation.
 */
static void
gen_allocate(int object, int stat)
{
  int i, ndim;
  int ast;
  int asd;
  int asym, dsym, dtype;
  ADSC *ad;
  int t;
  int nelem;
  int save_op_space;
  FtnRtlEnum rtlRtn;
  INT ty_val;

  if (A_TYPEG(object) == A_SUBSCR) {
    asd = A_ASDG(object);
    ast = A_LOPG(object);
    asym = find_pointer_variable(ast);
    dtype = DTYPEG(asym);
    dsym = DESCRG(asym);
    if (dsym) {
      gen_bnd_assn(object);
    }
    ndim = ASD_NDIM(asd);
    nelem = astb.i1;
    for (i = 0; i < ndim; i++) {
      int lw, up, triple, lb, ub, extnt;
      triple = ASD_SUBS(asd, i);
      lw = A_LBDG(triple);
      if (lw == 0) {
        lw = astb.i1;
      }
      up = A_UPBDG(triple);
      if (up == 0) {
        up = astb.i1;
      }
      t = mk_binop(OP_SUB, up, lw, DT_INT);
      t = mk_binop(OP_ADD, t, astb.i1, DT_INT);
      nelem = mk_binop(OP_MUL, nelem, t, DT_INT);
      if (SDSCG(asym) == 0 && !ALIGNG(asym)) {
        lb = ADD_LWAST(dtype, i);
        if (lb && A_TYPEG(lb) == A_ID && lb != lw) {
          /* put out assignment */
          put_string(SYMNAME(A_SPTRG(lb)));
          put_string(" = ");
          print_ast(lw);
        }
        ub = ADD_UPAST(dtype, i);
        if (up && A_TYPEG(ub) == A_ID && ub != up) {
          /* put out assignment */
          put_string(SYMNAME(A_SPTRG(ub)));
          put_string(" = ");
          print_ast(up);
        }
        extnt = ADD_EXTNTAST(dtype, i);
        if (extnt && A_TYPEG(extnt) == A_ID) {
          /* put out assignment */
          put_string(SYMNAME(A_SPTRG(extnt)));
          put_string(" = ");
          print_ast(mk_extent_expr(lw, up));
        }
      }
    }
  } else {
    ast = object;
    asym = find_pointer_variable(object);
    if (STYPEG(asym) == ST_ARRAY) {
      ad = AD_DPTR(DTYPEG(asym));
      nelem = AD_NUMELM(ad);
    } else
      nelem = astb.i1;
  }
  put_l_to_u("call ");
  rtlRtn = !ALLOCG(asym) ? RTE_ptr_alloca : RTE_alloca;
  put_string(mkRteRtnNm(rtlRtn));
  put_char('(');
  save_op_space = op_space;
  op_space = FALSE;
  print_ast(nelem); /* nelem */
  put_char(',');
  t = DTYPEG(asym);
  t = DTYG(t);
  ty_val = ty_to_lib[t];
  put_int(ty_val); /* kind */
  put_char(',');
  print_ast(size_ast(asym, DDTG(DTYPEG(asym)))); /* item length */
  put_char(',');
  if (stat)
    print_ast(stat); /* stat */
  else
    print_ast(astb.ptr0); /* 'null' stat */
  put_char(',');
  if (NO_PTR && XBIT(70, 8) && STYPEG(asym) == ST_MEMBER) {
    int mem;
    if (!F90POINTERG(asym) && POINTERG(asym) && PTROFFG(asym) &&
        STYPEG(PTROFFG(asym)) == ST_MEMBER) {
      print_ast_replaced(ast, asym, MIDNUMG(asym));
      put_char(',');
      print_ast_replaced(ast, asym, PTROFFG(asym));
    } else {
      print_ast(astb.ptr0); /* null pointer */
      put_char(',');
      print_ast_replaced(ast, asym, MIDNUMG(asym));
    }
    put_char(',');
    mem = find_member_base(DTYPEG(asym));
    if (mem <= NOSYM) {
      put_mem_string(ast, SYMNAME(asym));
    } else {
      put_string(SYMNAME(mem));
    }
  } else if (NO_PTR || /* no pointers in output */
             (NO_CHARPTR && DTYG(DTYPEG(asym)) == TY_CHAR) ||
             (NO_DERIVEDPTR && DTYG(DTYPEG(asym)) == TY_DERIVED)) {
    if (PTROFFG(asym)) {
      print_ast_replaced(ast, asym, MIDNUMG(asym));
      put_char(',');
      print_ast_replaced(ast, asym, PTROFFG(asym));
    } else {
      print_ast(astb.ptr0); /* null pointer */
      put_char(',');
      print_ast_replaced(ast, asym, MIDNUMG(asym));
    }
    put_char(',');
    put_mem_string(ast, SYMNAME(asym));
  } else {
    print_ast_replaced(ast, asym, MIDNUMG(asym));
    put_char(',');
    print_ast(astb.ptr0); /* null offset */
    put_char(',');
    print_ast(astb.ptr0); /* null base */
  }
  put_char(')');

  if (!F90POINTERG(asym) && POINTERG(asym) && DTY(DTYPEG(asym)) != TY_ARRAY) {
    /* assign the run-time type to the static descriptor created for
     * the scalar pointer.
     */
    print_ast_replaced(ast, asym, SDSCG(asym));
    put_string("(1) = ");
    put_int(ty_val); /* kind */
  }

  op_space = save_op_space;
}

/* If the output is 'standard' f77, need to convert the deallocate of an
 * object to a call to a run-time routine.  'object' is the id ast
 * representing the object; 'stat' is the id ast, not present if 0, of the
 * allocate status variable.
 * Note that the object is a pointer-based array; the associated pointer
 * variable is passed to the run-time routine.
 */
static void
gen_deallocate(int object, int stat, int asym, int passptr)
{
  assert(A_TYPEG(object) == A_ID || A_TYPEG(object) == A_MEM,
         "gen_deallocate:exp.id ast", object, 3);
  put_l_to_u("call ");
  if (passptr && MIDNUMG(asym) == 0) {
    passptr = 0;
  }
  if (passptr) {
    put_string(mkRteRtnNm(RTE_deallocx));
  } else {
    put_string(mkRteRtnNm(RTE_dealloca));
  }
  put_char('(');
  if (stat)
    print_ast(stat);
  else
    print_ast(astb.ptr0);
  put_char(',');
  if (NO_PTR && XBIT(70, 8) && STYPEG(asym) == ST_MEMBER) {
    int mem;
    mem = find_member_base(DTYPEG(asym));
    if (mem <= NOSYM) {
      put_mem_string(object, SYMNAME(asym));
    } else {
      put_string(SYMNAME(mem));
    }
    put_char('(');
    if (!F90POINTERG(asym) && POINTERG(asym) && PTROFFG(asym) &&
        STYPEG(PTROFFG(asym)) == ST_MEMBER) {
      print_ast_replaced(object, asym, PTROFFG(asym));
    } else {
      print_ast_replaced(object, asym, MIDNUMG(asym));
    }
    put_char(')');
  } else if (NO_PTR || /* no pointers in output */
             (NO_CHARPTR && DTYG(DTYPEG(asym)) == TY_CHAR) ||
             (NO_DERIVEDPTR && DTYG(DTYPEG(asym)) == TY_DERIVED)) {
    put_mem_string(object, SYMNAME(asym));
    put_char('(');
    if (PTROFFG(asym))
      print_ast_replaced(object, asym, PTROFFG(asym));
    else
      print_ast_replaced(object, asym, MIDNUMG(asym));
    put_char(')');
  } else if (passptr) {
    put_mem_string(object, SYMNAME(MIDNUMG(asym)));
  } else {
    put_mem_string(object, SYMNAME(asym));
  }
  put_char(')');
  if (POINTERG(asym) || passptr) {
    if (!NO_PTR || !XBIT(70, 8) || STYPEG(asym) != ST_MEMBER)
      gen_nullify(object, asym, passptr);
  }
}

static void
gen_nullify(int ast, int sym, int passptr)
{
  /* Pointer disassociation (statement):
   * nullify(pv)
   * call pghpf_nullify(pv, pv$sdsc)
   * pv:     base.
   * pv$sdsc:            pv's (new) static descriptor
   */
  FtnRtlEnum rtlRtn;

  if (F90POINTERG(sym)) {
    put_l_to_u("nullify( ");
    put_mem_string(ast, SYMNAME(sym));
    put_char(')');
    return;
  }
  if (STYPEG(sym) == ST_MEMBER) {
    /* do the nullify in-line */
    if (MIDNUMG(sym)) {
      print_ast_replaced(ast, sym, MIDNUMG(sym));
      put_string(" = 0");
    }
    if (PTROFFG(sym)) {
      print_ast_replaced(ast, sym, PTROFFG(sym));
      put_string(" = 0");
    }
    if (SDSCG(sym)) {
      print_ast_replaced(ast, sym, SDSCG(sym));
      put_string("(1) = 0");
    }
    return;
  }
  put_l_to_u("call ");
  if (DTYG(DTYPEG(sym)) == TY_CHAR) {
    rtlRtn = RTE_nullify_chara;
  } else if (passptr) {
    rtlRtn = RTE_nullifyx;
  } else {
    rtlRtn = RTE_nullify;
  }
  put_string(mkRteRtnNm(rtlRtn));
  put_char('(');

  if (passptr) {
    print_ast_replaced(ast, sym, MIDNUMG(sym));
  } else {
    put_mem_string(ast, SYMNAME(sym));
  }
  put_char(',');

  print_ast_replaced(ast, sym, SDSCG(sym));
  if (XBIT(70, 0x20)) {
    if (MIDNUMG(sym)) {
      put_char(',');
      print_ast_replaced(ast, sym, MIDNUMG(sym));
    }
    if (PTROFFG(sym)) {
      put_char(',');
      print_ast_replaced(ast, sym, PTROFFG(sym));
    }
  }
  put_char(')');
}

static void
put_string(const char *str)
{
  int len;

  len = strlen(str);
  check_len(len);
  strcpy(&lbuff[col], str);
  col += len;
}

static void
put_mem_string(int ast, const char *str)
{
  if (A_TYPEG(ast) == A_MEM) {
    print_ast(A_PARENTG(ast));
    put_string("%");
  }
  put_string(str);
}

static void
put_fstring(const char *str)
{
  int len;

  put_char('\'');
  /*
   * Can't use put_string() since may start 'str' on the next line leaving
   * 'spaces' after the quote.
   */
  len = strlen(str);
  while (len-- > 0)
    put_char(*str++);
  put_char('\'');
}

static void
put_char(char ch)
{
  check_len(1);
  lbuff[col] = ch;
  col++;
}

static void
put_const(int sptr)
{
  int len; /* length of character string */
  char b[64];
  char *from;
  int c;
  int dtype;
  int sptr2;

  dtype = DTYPEG(sptr);
  switch (DTY(dtype)) {
  case TY_WORD:
    sprintf(b, "z'%x'", CONVAL2G(sptr));
    put_l_to_u(b);
    return;
  case TY_DWORD:
    sprintf(b, "z'%x%08x'", CONVAL1G(sptr), CONVAL2G(sptr));
    put_l_to_u(b);
    return;
  case TY_BINT:
  case TY_SINT:
  case TY_INT:
    put_intkind(CONVAL2G(sptr), dtype);
    return;
  case TY_BLOG:
  case TY_SLOG:
  case TY_LOG:
  case TY_LOG8:
    put_logical(CONVAL2G(sptr), dtype);
    return;
  case TY_INT8:
    put_int8(sptr);
    return;
  case TY_REAL:
    if (NMPTRG(sptr)) {
      put_string(SYMNAME(sptr));
      return;
    }
    put_float(CONVAL2G(sptr));
    return;

  case TY_DBLE:
    if (NMPTRG(sptr)) {
      put_string(SYMNAME(sptr));
      return;
    }
    put_double(sptr);
    return;

  case TY_CMPLX:
    if (NMPTRG(sptr)) {
      put_string(SYMNAME(sptr));
      return;
    }
    put_char('(');
    put_float(CONVAL1G(sptr));
    put_char(',');
    put_float(CONVAL2G(sptr));
    put_char(')');
    return;

  case TY_DCMPLX:
    if (NMPTRG(sptr)) {
      put_string(SYMNAME(sptr));
      return;
    }
    put_char('(');
    put_const((int)CONVAL1G(sptr));
    put_char(',');
    put_const((int)CONVAL2G(sptr));
    put_char(')');
    return;

  case TY_HOLL:
    sptr2 = CONVAL1G(sptr);
    dtype = DTYPEG(sptr2);
    from = stb.n_base + CONVAL1G(sptr2);
    len = string_length(dtype);
    sprintf(b, "%d", len);
    put_string(b);
    b[0] = CONVAL2G(sptr); /* kind of hollerith - 'h', 'l', or 'r' */
    b[1] = '\0';
    put_l_to_u(b);
    while (len--) {
      c = *from++ & 0xff;
      put_char(c);
    }
    return;

  case TY_NCHAR:
    sptr = CONVAL1G(sptr); /* sptr to char string constant */
    dtype = DTYPEG(sptr);
    put_l_to_u("nc");
    FLANG_FALLTHROUGH;
  case TY_CHAR:
    from = stb.n_base + CONVAL1G(sptr);
    put_char('\'');
    len = string_length(dtype);
    while (len--)
      char_to_text(*from++);
    put_char('\'');
    return;

  case TY_PTR:
    strcpy(b, "address constant");
    break;

  default:
    strcpy(b, "bad_const_type");
  }

  put_string(b);
}

static void
put_int(INT val)
{
  char b[24];
  sprintf(b, "%d", val);
  put_string(b);
}

static void
put_intkind(INT val, int dtype)
{
  char b[30];
  INT vv;
  LOGICAL dokind;
  if (XBIT(57, 0x800)) {
    switch (DTY(dtype)) {
    case TY_BINT:
      vv = 0xffffff80;
      break;
    case TY_SINT:
      vv = 0xffff8000;
      break;
    case TY_INT:
      vv = 0x80000000;
      break;
    case TY_INT8:
      vv = 0;
      break;
    }
  }
  dokind = FALSE;
  if (DTY(DT_INT) != DTY(dtype)) {
    /* not default int - add _x to const */
    dokind = TRUE;
  }
  if (XBIT(57, 0x800) && val == vv) {
    sprintf(b, "%d", val + 1);
    if (dokind) {
      char *end;
      end = b + strlen(b);
      sprintf(end, "_%d", target_kind(dtype));
    }
    put_string("(");
    put_string(b);
    if (dokind) {
      sprintf(b, "-1_%d)", target_kind(dtype));
    } else {
      sprintf(b, "-1)");
    }
    put_string(b);
  } else {
    sprintf(b, "%d", val);
    if (dokind) {
      char *end;
      end = b + strlen(b);
      sprintf(end, "_%d", target_kind(dtype));
    }
    put_string(b);
  }
}

static void
put_int8(int sptr)
{
  char b[30];
  INT num[2];
  LOGICAL dokind;

  num[0] = CONVAL1G(sptr);
  num[1] = CONVAL2G(sptr);
  dokind = FALSE;
  if (DTY(DT_INT) != TY_INT8) {
    dokind = TRUE;
  }
  /* for most negative number, put out '(n-1)' */
  if (XBIT(57, 0x800) &&
      CONVAL1G(sptr) == (INT)(0x80000000) &&
      CONVAL2G(sptr) == 0) {
    num[1] = num[1] + 1;
    ui64toax(num, b, 22, 0, 10);
    if (dokind) {
      char *end;
      end = b + strlen(b);
      sprintf(end, "_%d", target_kind(DT_INT8));
    }
    put_string("(");
    put_string(b);
    if (dokind) {
      sprintf(b, "-1_%d)", target_kind(DT_INT8));
    } else {
      sprintf(b, "-1)");
    }
    put_string(b);
  } else {
    ui64toax(num, b, 22, 0, 10);
    if (dokind) {
      char *end;
      end = b + strlen(b);
      sprintf(end, "_%d", target_kind(DT_INT8));
    }
    put_string(b);
  }
}

static void
put_logical(LOGICAL val, int dtype)
{
  char b[20];
  if (val & 1)
    sprintf(b, ".true.");
  else
    sprintf(b, ".false.");
  if (DTY(dtype) != DT_LOG) {
    char *bb;
    for (bb = b; *bb; ++bb)
      ;
    sprintf(bb, "_%d", target_kind(dtype));
  }
  put_string(b);
}

static void
put_float(INT val)
{
  char b[64];
  char *start;
  char *end;
  int i;
  char *exp;
  int expw;

  /* FIXME double cast is done to silence the warning, this needs to be
   * revisited!  cprintf (our cprintf, not system routine) takes a pointer but
   * in this particular case uses is as an integer
   */
  cprintf(b, "%.10e", (INT *)((BIGINT)val));
  for (start = b; *start == ' '; start++) /* skip leading blanks */
    ;
  /* only leave the sign if it's '-' */
  if (*start == '+')
    start++;

  /* locate beginning of exponent */
  exp = &b[strlen(b) - 1];
  expw = -1; /* width of exponent less 'E' and the sign */
  while (*exp != 'E' && *exp != 'e' && *exp != 'D' && *exp != 'd') {
    if (exp <= start) {
      /* output from cprintf is [-]INF */
      if (*start == '-')
        put_char('-');
      put_string("1e+39");
      return;
    }
    exp--;
    expw++;
  }

  i = (exp - b) - 1; /* last decimal digit */
                     /*
                      * omit trailing 0's; don't omit digit after the decimal point.
                      */
  while (b[i] == '0' && i > 3)
    i--;
  end = &b[i + 1];
  /* exp locates 'E' */
  *end++ = 'e';
  if (*++exp == '-') /* sign */
    *end++ = '-';
  if (expw == 2) {
    if (*++exp != '0')
      *end++ = *exp;
    *end++ = *++exp;
  } else {
    while (expw--)
      *end++ = *++exp;
  }
  if (DTY(DT_REAL) != TY_REAL) {
    /* f90 output */
    *end++ = '_';
    sprintf(end, "%d", target_kind(DT_REAL4));
  } else
    *end = '\0';
  put_string(start);
}

static void
put_double(int sptr)
{
  INT num[2];
  char b[64];
  char *start;
  char *end;
  char *exp;
  int expw;
  int i;

  num[0] = CONVAL1G(sptr);
  num[1] = CONVAL2G(sptr);

  /* warning:  there may be 2 or digits in the exponent -- D<sign>dd or
   *           D<sign>ddd.
   */

  if (XBIT(49, 0x40000)) /* C90 */
    cprintf(b, "%.15ld", num);
  else
    cprintf(b, "%.17ld", num);

  for (start = b; *start == ' '; start++) /* skip leading blanks */
    ;
  /* only leave the sign if it's '-' */
  if (*start == '+')
    start++;

  /* locate beginning of exponent */
  exp = &b[strlen(b) - 1];
  expw = -1; /* width of exponent less 'D' and the sign */
  while (*exp != 'E' && *exp != 'e' && *exp != 'D' && *exp != 'd') {
    if (exp <= start) {
      /* output from cprintf is [-]INF */
      if (*start == '-')
        put_char('-');
      put_string("1d+309");
      return;
    }
    exp--;
    expw++;
  }

  i = (exp - b) - 1; /* last decimal digit */
                     /*
                      * omit trailing 0's; don't omit digit after the decimal point.
                      */
  while (b[i] == '0' && i > 3)
    i--;
  end = &b[i + 1];
  /* exp locates 'D' */
  if (DTY(DT_REAL) == TY_DBLE && XBIT(49, 0x800000))
    /* change 'd' to 'e' only if default real is double precision for
     * the cray systems.
     */
    *end++ = 'e';
  else
    *end++ = 'd';
  if (*++exp == '-') /* sign */
    *end++ = '-';
  if (expw == 2) {
    if (*++exp != '0')
      *end++ = *exp;
    *end++ = *++exp;
  } else {
    while (expw--)
      *end++ = *++exp;
  }
  *end = '\0';
  put_string(start);
}

/*
 * emit a character with consideration given to the ', escape sequences,
 * unprintable characters, etc.
 */
static void
char_to_text(int ch)
{
  int c;
  char b[8];

  c = ch & 0xff;
  if (c == '\\' && !XBIT(124, 0x40)) {
    put_char('\\');
    put_char('\\');
  } else if (c == '\'') {
    put_char('\'');
    put_char('\'');
  } else if (c >= ' ' && c <= '~')
    put_char(c);
  else if (XBIT(52, 0x10)) {
    put_char(c);
  } else if (c == '\n') {
    put_char('\\');
    put_char('n');
  } else if (c == '\t') {
    put_char('\\');
    put_char('t');
  } else if (c == '\v') {
    put_char('\\');
    put_char('v');
  } else if (c == '\b') {
    put_char('\\');
    put_char('b');
  } else if (c == '\r') {
    put_char('\\');
    put_char('r');
  } else if (c == '\f') {
    put_char('\\');
    put_char('f');
  } else {
    /* Mask off 8 bits worth of unprintable character */
    sprintf(b, "\\%03o", c);
    put_string(b);
  }
}

/* emit name when it's known to contain uppercase letters;
 * convert upper to lower if necessary.
 */
static void
put_u_to_l(const char *name)
{
  char ch;

  if (flg.ucase)
    put_string(name);
  else {
    check_len(strlen(name));
    while ((ch = *name++)) {
      ch &= 0xff;
      if (isupper(ch))
        ch += 32;
      lbuff[col] = ch;
      col++;
    }
  }
}

/* emit name when it's known to contain lowercase letters, e.g., keywords.
 * TBD - convert lower to upper if necessary.
 */
static void
put_l_to_u(const char *name)
{
  put_string(name);
}

static int just_did_sharpline = 0;

static void
write_next_line(void)
{
  lbuff[col] = '\0';
  fprintf(outfile, "%s\n", lbuff);
  just_did_sharpline = 0;
  col = 0;
}

static void
check_len(int len)
{
  if ((len + col) > max_col) {
    write_next_line();
    ++continuations;
  }
}

static char *
label_name(int lab)
{
  char *nm;
  char lbuff[8];
  static int lbavail = 99999;

  nm = SYMNAME(lab);
  if (CCSYMG(lab)) {
    /* compiler-created label - ensure that its number doesn't conflict
     * with a user label.
     */
    int lb;

    if (SYMLKG(lab))
      /* one is already created */
      lb = SYMLKG(lab);
    else {
      lbuff[0] = '.';            /* user label begins with '.L' */
      strcpy(&lbuff[1], nm + 1); /* copy 'L' followed by the digits */
                                 /*
                                  * search for a label which doesn't conflict.
                                  */
      while (TRUE) {
        if (lookupsym(lbuff, 7) == 0)
          break;
        sprintf(&lbuff[2], "%05d", lbavail--);
      }
      lb = getsym(lbuff, 7);
      STYPEP(lb, ST_LABEL);
      SYMLKP(lab, lb);
    }
    nm = SYMNAME(lb);
  }
  nm += 2; /* skip past .L */
  while (*nm == '0')
    nm++; /* skip over leading 0's */
  return nm;
}

/* subp is the sptr of subprogram */
static void
print_header(int subp)
{
  int dscptr;
  int arg;
  int i;

  print_sname(subp);
  put_char('(');
  if ((i = PARAMCTG(subp))) {
    dscptr = DPDSCG(subp);
    while (TRUE) {
      arg = aux.dpdsc_base[dscptr];
      if (arg)
        put_string(SYMNAME(arg));
      else
        put_char('*'); /* alternate return specifier */
      if (--i == 0)
        break;
      put_char(',');
      dscptr++;
    }
  }
  put_char(')');
}

/** \brief Add parameters in the order in which they were declared.
 */
void
add_param(int sptr)
{
  _A_Q *q;

  if (sem.which_pass == 0)
    return;
  if (VAXG(sptr)) {
    if (A_TYPEG(CONVAL2G(sptr)) == A_CNST)
      q = &vx_params.q;
    else
      q = &vx_params.q_e;
  } else {
    if (A_TYPEG(CONVAL2G(sptr)) == A_CNST)
      q = &params.q;
    else
      q = &params.q_e;
  }

  if (q->first == 0)
    q->first = sptr;
  else
    SYMLKP(q->last, sptr);
  q->last = sptr;
  SYMLKP(sptr, 0);
  ENDP(sptr, 0);
}

/** \brief Since a separate list is created for each parameter combination of
    ansi-/vax- style and constant/non-constant ast, it is necessary to
    mark where in the list the contributions from a each parameter statement
    ends.
 */
void
end_param(void)
{
  static _A_Q *q[] = {&params.q, &params.q_e, &vx_params.q, &vx_params.q_e};
  int i;

  for (i = 0; i < 4; i++) {
    if (q[i]->first)
      ENDP(q[i]->last, 1);
  }
}

static void
pghpf_entry(int func)
{
  INT fl;

  if (!XBIT(49, 0x1000))
    return;

  /* pghpf_function_entry(line,nlines,function,file) */

  put_l_to_u("call ");
  put_string(mkRteRtnNm(RTE_function_entrya));
  put_char('(');
  fl = FUNCLINEG(func);
  put_int(fl);
  put_char(',');
  put_int(ENDLINEG(func) - fl + 1);
  put_char(',');
  put_fstring(SYMNAME(func));
  put_char(',');
  put_fstring(gbl.src_file);
  put_char(')');
}

void
dbg_print_ast(int ast, FILE *fil)
{
  int save_max_col;

  col = 0;
  if (fil == NULL)
    fil = stderr;
  outfile = fil;
  save_max_col = max_col;
  max_col = 299;
  init_line();
  indent = 0;
  ast_is_comment = TRUE;
  print_ast(ast);
  if (col != 0)
    write_next_line();
  ast_is_comment = FALSE;
  max_col = save_max_col;
}

void
dbg_print_stmts(FILE *f)
{
  int std;
  int ast;

  if (f == NULL)
    f = stderr;
  for (std = STD_NEXT(0); std; std = STD_NEXT(std)) {
    ast = STD_AST(std);
    dbg_print_ast(ast, f);
  }
}

void
printast(int ast)
{
  if (gbl.dbgfil == NULL) {
    outfile = stderr;
  } else {
    outfile = gbl.dbgfil;
  }
  indent = 0;
  col = 0;
  ast_is_comment = TRUE;
  print_ast(ast);
  lbuff[col] = '\0';
  fprintf(outfile, "%s", lbuff);
  ast_is_comment = FALSE;
}
