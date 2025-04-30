/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* transutl.c - PGHPF transformation utilities */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "soc.h"
#include "semant.h"
#include "ast.h"
#include "gramtk.h"
#include "hpfutl.h"
#include "optimize.h"
#include "hlvect.h"
int
mk_ceil_div(int num, int den)
{
  /* compute CD(num, den) = (num + den - 1) / den */
  int one = mk_cval(1, DT_INT);
  int a;

  if (den == 0 || den == one)
    return num;
  a = opt_binop(OP_ADD, num, den, DT_INT);
  a = opt_binop(OP_SUB, a, one, DT_INT);
  a = opt_binop(OP_DIV, a, den, DT_INT);
  return opt_ast(a);
}

int
opt_binop(int opc, int op1, int op2, int dtype)
{
  int ast;
  ast = mk_binop(opc, op1, op2, dtype);
  ast = dd_symbolic(ast);
  freearea(HLV_AREA1);
  return ast;
}

int
opt_unop(int opc, int op, int dtype)
{
  return mk_unop(opc, op, dtype);
}

int
opt_ast(int a)
{
/* optimize an AST.  Recursively search for subtrees of '+' nodes
 * and try to combine constants.  Turn '-' nodes into '+' 'U-' nodes.
 * May want to add distribution later.
 */
  return a;
}

int
mk_size(int low, int high)
{
  int t;

  if (low == 0)
    low = mk_cval(1, DT_INT);
  if (high == 0)
    high = low;
  if (A_TYPEG(low) == A_CNST && A_TYPEG(high) == A_CNST) {
    t = get_int_cval(A_SPTRG(high)) - get_int_cval(A_SPTRG(low)) + 1;
    return mk_cval(t, DT_INT);
  }
  t = opt_binop(OP_ADD, opt_binop(OP_SUB, high, low, DT_INT),
                mk_cval(1, DT_INT), DT_INT);
  return opt_ast(t);
}

int
search_basearr(int ast)
{
  /* search the ast for the symbol describing the base array.  For
   * now, ast can be either subscript or ID.  Eventually, we will
   * need to handle records */

  if (A_TYPEG(ast) == A_ID)
    return A_SPTRG(ast);
  if (A_TYPEG(ast) == A_SUBSCR)
    return search_basearr(A_LOPG(ast));
  interr("search_basearr: can't find basearr", ast, 3);
  return 0;
}
