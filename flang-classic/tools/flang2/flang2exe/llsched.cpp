/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief scheduling routines for LLVM Code Generator
 */

#include "llsched.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "cgllvm.h"
#include "ili.h"
#include <stdlib.h>

static INSTR_LIST **matrix_dg;
static INSTR_LIST *last_instr;
static int irank;
static int size_dg;
static int srank_dg;

static bool
init_sched_graph(int size, int srank)
{
  size_dg = size;
  srank_dg = srank;
  matrix_dg = (INSTR_LIST**)realloc(matrix_dg, size * size *
                                    sizeof(INSTR_LIST));
  if (matrix_dg)
    memset(matrix_dg, 0, size * size * sizeof(INSTR_LIST));
  return (matrix_dg != NULL);
}

static void
add_successor(INSTR_LIST *instr, INSTR_LIST *succ)
{
  int i, j;

  /* avoid creating circular dependency */
  if (instr == succ)
    return;
  if (succ->flags & ROOTDG)
    return;
  i = instr->rank - srank_dg;
  j = succ->rank - srank_dg;
  matrix_dg[size_dg * i + j] = succ;
}

static bool
build_same_base_nme(int nme1, int nme2, int *res_nme)
{
  int nme;
  if (nme1 && nme2) {
    if (NME_TYPE(nme1) != NME_TYPE(nme2))
      return false;

    switch (NME_TYPE(nme1)) {
    default:
      break;
    case NT_VAR:
      if (NME_SYM(nme1) != NME_SYM(nme2) &&
          STYPEG(NME_SYM(nme1)) == STYPEG(NME_SYM(nme2)) &&
          DTY(DTYPEG(NME_SYM(nme1))) == DTY(DTYPEG(NME_SYM(nme2)))) {
        *res_nme = addnme(NT_VAR, NME_SYM(nme1), 0, 0);
        return true;
      }
      break;
    case NT_ARR:
    case NT_MEM:
    case NT_IND:
      if (build_same_base_nme(NME_NM(nme1), NME_NM(nme2), &nme)) {
        *res_nme = add_arrnme(NME_TYPE(nme2), NME_SYM(nme2), nme,
                              NME_CNST(nme2), NME_SUB(nme2), NME_INLARR(nme2));
        return true;
      }
      break;
    }
  }
  return false;
}

int
enhanced_conflict(int nme1, int nme2)
{
  int c;
  int nme3;

  c = conflict(nme1, nme2);
  if (XBIT(183, 0x80) && c != NOCONFLICT && c != SAME) {
    if (build_same_base_nme(nme1, nme2, &nme3))
      c = conflict(nme1, nme3);
  }
  return c;
}

static void
build_idep_graph(INSTR_LIST *iroot, INSTR_LIST *cur_instr)
{
  INSTR_LIST *instr;
  int instcount;
  int ilix, c;
  bool first_load, has_pred;
  OPERAND *operand;

  has_pred = false;
  switch (cur_instr->i_name) {
  default:
    break;
  case I_LOAD:
    assert(cur_instr->tmps, "build_dep_graph():missing tmps for load instr ", 0,
           ERR_Fatal);
    instr = cur_instr->prev;
    ilix = cur_instr->ilix;
    first_load = true;
    instcount = 0;
    while (instr && instr != iroot) {
      instcount++;
      switch (instr->i_name) {
      default:
        break;
      case I_STORE:
        /* if conflicting store then add this load to its successor's list */
        if ((ilix == 0) || (instr->ilix == 0) ||
            (IL_TYPE(ILI_OPC(instr->ilix)) != ILTY_STORE) ||
            (ILI_OPND(ilix, 1) == ILI_OPND(instr->ilix, 2))) {
          has_pred = true;
          add_successor(instr, cur_instr);
        } else {
          c = enhanced_conflict(ILI_OPND(ilix, 2), ILI_OPND(instr->ilix, 3));
          if (c == SAME || (flg.depchk && c != NOCONFLICT)) {
            has_pred = true;
            add_successor(instr, cur_instr);
          }
        }
        break;
      }
      instr = instr->prev;
    }
    break;
  case I_STORE:
    instr = cur_instr->prev;
    ilix = cur_instr->ilix;
    first_load = true;
    while (instr && instr != iroot) {
      switch (instr->i_name) {
      default:
        break;
      case I_STORE:
        /* if conflicting store then add this store to its successor's list */
        if ((ilix == 0) || (instr->ilix == 0) ||
            (IL_TYPE(ILI_OPC(instr->ilix)) != ILTY_STORE) ||
            (ILI_OPND(ilix, 2) == ILI_OPND(instr->ilix, 2))) {
          has_pred = true;
          add_successor(instr, cur_instr);
        } else {
          c = enhanced_conflict(ILI_OPND(ilix, 3), ILI_OPND(instr->ilix, 3));
          if (c == SAME || (flg.depchk && c != NOCONFLICT)) {
            has_pred = true;
            add_successor(instr, cur_instr);
          }
        }
        break;
      case I_LOAD:
        /* if conflicting store then add this store to its successor's list */
        if ((ilix == 0) || (instr->ilix == 0) ||
            (IL_TYPE(ILI_OPC(instr->ilix)) != ILTY_STORE) ||
            (ILI_OPND(ilix, 2) == ILI_OPND(instr->ilix, 1))) {
          has_pred = true;
          add_successor(instr, cur_instr);
        } else {
          c = enhanced_conflict(ILI_OPND(ilix, 3), ILI_OPND(instr->ilix, 2));
          if (c == SAME || (flg.depchk && c != NOCONFLICT)) {
            has_pred = true;
            add_successor(instr, cur_instr);
          }
        }
        break;
      }
      instr = instr->prev;
    }
    break;
  }
  operand = cur_instr->operands;
  while (operand) {
    if (operand->ot_type == OT_TMP) {
      instr = cur_instr->prev;
      while (instr) {
        if (operand->tmps == instr->tmps) {
          has_pred = true;
          add_successor(instr, cur_instr);
          break;
        }
        if (instr == iroot)
          break;
        instr = instr->prev;
      }
    }
    operand = operand->next;
  }
  if (!has_pred)
    add_successor(iroot, cur_instr);
}

static bool
is_lltype_interesting(LL_Type *llt)
{
  if (llt) {
    switch (llt->data_type) {
    case LL_VECTOR:
    case LL_FLOAT:
    case LL_DOUBLE:
      return true;
    default:
      break;
    }
  }
  return false;
}

static INSTR_LIST *
build_block_idep_graph(INSTR_LIST *istart, bool *success)
{
  INSTR_LIST *instr, *bbinstr;
  int inst_count = 0;
  int interesting_stores = 0;
  int interesting_loads = 0;
  int interesting_instrs = 0;

  instr = istart->next;
  *success = false;
  while (instr) {
    instr->rank = irank++;
    inst_count++;
    switch (instr->i_name) {
    case I_LOAD:
      if (interesting_stores && is_lltype_interesting(instr->ll_type))
        interesting_loads++;
      break;
    case I_STORE:
      if (is_lltype_interesting(instr->operands->ll_type))
        interesting_stores++;
      break;
    case I_CALL:
    case I_SW:
    case I_RET:
    case I_BR:
    case I_NONE:
    case I_FDIV:
    case I_FPTRUNC:
    case I_FPEXT:
      if (XBIT(183, 0x100) ||
          ((inst_count > 50) && interesting_instrs &&
           (interesting_loads > ((interesting_stores * 3) / 2)))) {
        bbinstr = istart->next;
        *success = init_sched_graph(inst_count, istart->rank);
        if (*success) {
          instr->flags |= ROOTDG;
          while (bbinstr && (bbinstr != instr->next)) {
            build_idep_graph(istart, bbinstr);
            bbinstr = bbinstr->next;
          }
        }
      }
      return instr;
    default:
      if (BINOP(instr->i_name) || BITOP(instr->i_name) ||
          CONVERT(instr->i_name) || PICALL(instr->i_name))
        if (is_lltype_interesting(instr->ll_type)) {
          interesting_instrs++;
        }
    }
    instr = instr->next;
  }
  return instr;
}

void
sched_block_breadth_first(INSTR_LIST *istart, int level)
{
  int i, j, k;
  INSTR_LIST *succ, *pred, *entry;

  i = istart->rank - srank_dg;

  entry = last_instr;
  /* Schedule GEP/LOAD/BITCAST first */
  for (j = 0; j < size_dg; j++) {
    succ = matrix_dg[i * size_dg + j];
    if (succ) {
      switch (succ->i_name) {
      default:
        break;
      case I_GEP:
      case I_BITCAST:
      case I_LOAD:
        matrix_dg[i * size_dg + j] = NULL;
        pred = NULL;
        for (k = 0; k < size_dg; k++) {
          pred = matrix_dg[k * size_dg + j];
          if (pred)
            break;
        }
        if (pred == NULL) {
          last_instr->next = succ;
          succ->prev = last_instr;
          last_instr = succ;
          last_instr->next = NULL;
        }
        break;
      }
    }
  }

  /* Schedule any instructions but stores */
  for (j = 0; j < size_dg; j++) {
    succ = matrix_dg[i * size_dg + j];
    if (succ) {
      if (succ->i_name != I_STORE) {
        matrix_dg[i * size_dg + j] = NULL;
        pred = NULL;
        for (k = 0; k < size_dg; k++) {
          pred = matrix_dg[k * size_dg + j];
          if (pred)
            break;
        }
        if (pred == NULL) {
          last_instr->next = succ;
          succ->prev = last_instr;
          last_instr = succ;
          last_instr->next = NULL;
        }
      }
    }
  }

  /* Schedule stores */
  for (j = 0; j < size_dg; j++) {
    succ = matrix_dg[i * size_dg + j];
    if (succ) {
      if (succ->i_name == I_STORE) {
        matrix_dg[i * size_dg + j] = NULL;
        pred = NULL;
        for (k = 0; k < size_dg; k++) {
          pred = matrix_dg[k * size_dg + j];
          if (pred)
            break;
        }
        if (pred == NULL) {
          last_instr->next = succ;
          succ->prev = last_instr;
          last_instr = succ;
          last_instr->next = NULL;
        }
      }
    }
  }

  succ = entry->next;
  while (succ) {
    sched_block_breadth_first(succ, level + 1);
    succ = succ->next;
  }
}

void
check_circular_dep(INSTR_LIST *istart)
{
  int i, j;
  INSTR_LIST *succ;

  i = istart->rank - srank_dg;
  if (istart->flags & INST_VISITED) {
    printf(" CIRCULAR DEPENDENCY !\n");
    return;
  }
  istart->flags |= INST_VISITED;
  for (j = 0; j < size_dg; j++) {
    succ = matrix_dg[i * size_dg + j];
    if (succ) {
      printf("i%d -> ", istart->rank);
      check_circular_dep(succ);
    }
  }
  printf("\n");
  istart->flags &= ~INST_VISITED;
}

void
sched_block(INSTR_LIST *istart, INSTR_LIST *iend)
{
  if (istart && istart->next != iend) {
    last_instr = istart;
    last_instr->next = NULL;
    sched_block_breadth_first(istart, 0);
    last_instr->next = iend;
    if (iend != NULL)
      iend->prev = last_instr;
  }
}

void
sched_instructions(INSTR_LIST *istart)
{
  INSTR_LIST *instr;
  bool dep_graph_created = false;
  irank = 1;
  istart->flags |= ROOTDG;

  while (istart) {
    instr = build_block_idep_graph(istart, &dep_graph_created);
    if (dep_graph_created)
      sched_block(istart, instr);
    istart = instr;
  }
}
