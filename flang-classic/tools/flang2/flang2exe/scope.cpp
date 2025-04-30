/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Functions for dealing with lexical scopes and the lifetimes
   of scoped variables.


   A lexical scope is represented by an ST_BLOCK symbol table entry. The
   scope's ENCLFUNC field points to either the enclosing scope or the
   containing function. Scopes have two pairs of start/end labels:

   STARTLAB / ENDLAB: These labels are only inserted when generating debug
   info. The are tagged as volatile so they won't be removed, and they can be
   used to generate complete DWARF lexical scope information.

   BEGINSCOPELAB / ENDSCOPELAB: These labels are inserted in-stream as ilts
   with an IL_LABEL instruction. The corresponding ST_LABEL symbol table
   entries are tagged with the corresponding BEGINSCOPE / ENDSCOPE flags, and
   the ENCLFUNC field points back to the ST_BLOCK entry.

   Every local variable has a scope indicated by its ENCLFUNC field. Variables
   with function scope point to the function's symbol table entry instead of an
   ST_BLOCK entry.
*/

#include "scope.h"
#include "error.h"
#include "global.h"
#include "ilm.h"
#include "ilmtp.h"
#include "ili.h"
#include "go.h"
#include "scope.h"
#include "flang/ADT/hash.h"
#include "symfun.h"

/* For addlabel(). */
#include "semant.h"

#define VALIDSYM(sptr) ((sptr) > NOSYM && (sptr) < stb.stg_avail)

/**
   \brief Is sptr a symbol table entry that can be the root of a scope
   tree?  For C, this is functions. For Fortran, this includes entries
   and procedures.
 */
static bool
is_scope_root(SPTR sptr)
{
  switch (STYPEG(sptr)) {
  case ST_PROC:
  case ST_ENTRY:
    return true;
  default:
    return false;
  }
}

/**
   \brief Return true if the scope 'outer' contains the scope 'inner'.

   Both outer and inner must be valid ST_BLOCK or ST_FUNC symbol table
   entries.
 */
bool
scope_contains(SPTR outer, SPTR inner)
{
  ICHECK(VALIDSYM(outer) &&
         (is_scope_root(outer) || STYPEG(outer) == ST_BLOCK));
  ICHECK(VALIDSYM(inner) &&
         (is_scope_root(inner) || STYPEG(inner) == ST_BLOCK));

  while (STYPEG(inner) == ST_BLOCK) {
    if (inner == outer)
      return true;
    inner = ENCLFUNCG(inner);
  }

  return inner == outer;
}

/**
   \brief Return true if label_sptr is a scope label.
 */
bool
is_scope_label(int label_sptr)
{
  if (!VALIDSYM(label_sptr) || STYPEG(label_sptr) != ST_LABEL)
    return false;

  return BEGINSCOPEG(label_sptr) || ENDSCOPEG(label_sptr);
}

/**
   \brief Check if ilix is an IL_LABEL instruction referring to a
   scope label.
 */
bool
is_scope_label_ili(int ilix)
{
  return ILI_OPC(ilix) == IL_LABEL && is_scope_label(ILI_OPND(ilix, 1));
}

/**
   \brief Check if the bihx basic block contains nothing but scope
   labels (or is empty).
 */
bool
is_scope_labels_only_bih(int bihx)
{
  int iltx;

  for (iltx = BIH_ILTFIRST(bihx); iltx; iltx = ILT_NEXT(iltx)) {
    if (!is_scope_label_ili(ILT_ILIP(iltx)))
      return false;
  }
  return true;
}

/**
   \brief Verify the integrity of an ST_BLOCK symbol table entry and
   its associated labels.
 */
static void
verify_block(int block_sptr)
{
  int lab;

  ICHECK(VALIDSYM(block_sptr));
  ICHECK(STYPEG(block_sptr) == ST_BLOCK);

  /* STARTLAB and ENDLAB are not required to be present. */
  if (STARTLABG(block_sptr)) {
    lab = STARTLABG(block_sptr);
    ICHECK(VALIDSYM(lab));
    ICHECK(STYPEG(lab) == ST_LABEL);
    ICHECK(ENCLFUNCG(lab) == block_sptr);

    /* These flags should go on the BEGINSCOPELAB / ENDSCOPELAB
       labels, not here. */
    ICHECK(!BEGINSCOPEG(lab));
    ICHECK(!ENDSCOPEG(lab));
    /* FIXME: Check ILIBLK. */
  }

  if (ENDLABG(block_sptr)) {
    lab = ENDLABG(block_sptr);
    ICHECK(VALIDSYM(lab));
    ICHECK(STYPEG(lab) == ST_LABEL);
    ICHECK(ENCLFUNCG(lab) == block_sptr);
    ICHECK(!BEGINSCOPEG(lab));
    ICHECK(!ENDSCOPEG(lab));
    /* FIXME: Check ILIBLK. */
  }

  /* All blocks should have delineating labels. */
  lab = BEGINSCOPELABG(block_sptr);
  ICHECK(VALIDSYM(lab));
  ICHECK(STYPEG(lab) == ST_LABEL);
  ICHECK(ENCLFUNCG(lab) == block_sptr);
  ICHECK(BEGINSCOPEG(lab));
  ICHECK(!ENDSCOPEG(lab));
  /* FIXME: Check ILIBLK. */

  lab = ENDSCOPELABG(block_sptr);
  ICHECK(VALIDSYM(lab));
  ICHECK(STYPEG(lab) == ST_LABEL);
  ICHECK(ENCLFUNCG(lab) == block_sptr);
  ICHECK(!BEGINSCOPEG(lab));
  ICHECK(ENDSCOPEG(lab));
  /* FIXME: Check ILIBLK. */
}

static bool
is_local_variable(int sptr)
{
  switch (STYPEG(sptr)) {
  case ST_PLIST:
  case ST_VAR:
  case ST_ARRAY:
  case ST_STRUCT:
  case ST_UNION:
    switch (SCG(sptr)) {
    case SC_LOCAL:
      /* Somebody who understands Fortran should probably revisit this
         and check DINITG, SAVEG etc. */
      return true;
    /* Ignore SC_DUMMY. It's going to have function scope anyway. */
    default:
      return false;
    }
  default:
    return false;
  }
}

/**
   \brief Collect the set of locals referenced by the ILI tree starting at ilix.

   Loads and stores to local variables are always going to use an ACON
   to compute the address, so just find all the symbols referenced by
   ACONs.
 */
static void
collect_referenced_locals(hashset_t symbs, int ilix)
{
  ILI_OP opc = ILI_OPC(ilix);

  if (opc == IL_ACON) {
    /* ACON opnd 1 is an ST_CONST sptr with DT_CPTR type. */
    int sptr = CONVAL1G(ILI_OPND(ilix, 1));
    if (sptr && is_local_variable(sptr))
      hashset_replace(symbs, INT2HKEY(sptr));
  } else {
    int i;
    for (i = IL_OPRS(opc); i > 0; i--)
      if (IL_ISLINK(opc, i))
        collect_referenced_locals(symbs, ILI_OPND(ilix, i));
  }
}

/* Info about a single open scope. */
struct scope_info {
  /* ST_BLOCK for scope, or ST_FUNC/ST_PROC/ST_ENTRY for the function-level
     scope. */
  SPTR block;
  /* The IL_LABEL ilt that closes this scope. */
  int closing_ilt;
};

/* A stack of open scopes. */
struct scope_stack {
  /* scope[0..curr] are in use. */
  int curr;
  /* scope[0..outer] are perfectly nested. */
  int outer;
  /* Allocated entries in scope. */
  int allocated;
  struct scope_info *scope;
  /* scope[outer].block from before closed scopes were popped. */
  SPTR prev_top;
};

/**
   \brief Initialize a scope stack with func as the outermost scope.
 */
static void
init_scope_stack(struct scope_stack *ss, SPTR func)
{
  ss->curr = ss->outer = 0;
  ss->allocated = 100;
  NEW(ss->scope, struct scope_info, ss->allocated);
  memset(&ss->scope[0], 0, sizeof(ss->scope[0]));
  ss->scope[0].block = func;
  ss->prev_top = SPTR_NULL;
}

static void
push_block(struct scope_stack *ss, SPTR block)
{
  ++ss->curr;
  NEED(ss->curr + 1, ss->scope, struct scope_info, ss->allocated,
       ss->allocated + 100);
  memset(&ss->scope[ss->curr], 0, sizeof(ss->scope[0]));
  ss->scope[ss->curr].block = block;
  dbgprintf("  Pushing scope %d (%s) at level %d: enclfunc=%d (%s)\n", block,
            SYMNAME(block), ss->curr, ENCLFUNCG(block),
            SYMNAME(ENCLFUNCG(block)));
}

static void
pop_block(struct scope_stack *ss)
{
  dbgprintf("  Popping scope %d (%s) at level %d\n", ss->scope[ss->curr].block,
            SYMNAME(ss->scope[ss->curr].block), ss->curr);
  ICHECK(ss->curr > 0);
  if (--ss->curr < ss->outer)
    ss->outer = ss->curr;
}

/**
   \brief Check labels opening scopes in bih, and collect all symbols
   referenced in this block.

   Add opened scopes to ss.
 */
static void
verify_beginscopes(int bih, hashset_t locals, struct scope_stack *ss)
{
  int ilt;

  /* All of the scopes pushed in this block must be sub-scopes of the
     current top. */
  SPTR outer_scope = ss->scope[ss->curr].block;

  for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
    int ilix = ILT_ILIP(ilt);
    SPTR lab, blk;

    if (ILI_OPC(ilix) != IL_LABEL) {
      collect_referenced_locals(locals, ilix);
      continue;
    }

    /* This is an IL_LABEL ilt. */
    lab = ILI_SymOPND(ilix, 1);
    ICHECK(VALIDSYM(lab) && STYPEG(lab) == ST_LABEL);
    if (!BEGINSCOPEG(lab))
      continue;

    /* This is a scope-opening label. */
    blk = ENCLFUNCG(lab);
    verify_block(blk);
    ICHECK(BEGINSCOPELABG(blk) == lab);

    /* If multiple scopes are pushed in the same BIH, we'll allow any
       order. They all have to be proper sub-scopes of outer_scope,
       though. */
    push_block(ss, blk);
    ICHECK((blk != outer_scope) && "Can't open the same scope twice");
    ICHECK(scope_contains(outer_scope, blk) &&
           "New scope is not properly nested");
  }
}

/**
   \brief Verify a local that was referenced in the current function.

   This function conforms to the prototype expected by hashset_iterate.
 */
static void
verify_local(hash_key_t key, void *context)
{
  int sptr = HKEY2INT(key);
  struct scope_stack *ss = (struct scope_stack *)context;
  SPTR encl = ENCLFUNCG(sptr);
  int i;

/* The Fortran frontend doesn't bother setting ENCLFUNC on local
   variables.  Fortran doesn't have scopes anyway, except through
   inlining. */
  if (!encl)
    return;

  /* Compiler-created scalar temporaries don't always have associated
     scopes.  They will be treated as function-scope. */
  if (!encl && CCSYMG(sptr)) {
    dbgprintf("  Referenced local compiler-created temporary with no "
              "scope: %d (%s)\n",
              sptr, SYMNAME(sptr));
    return;
  }

  dbgprintf("  Referenced local %d (%s) from scope %d. At block %d level "
            "%d-%d, prev %d.\n",
            sptr, SYMNAME(sptr), encl, ss->scope[ss->curr].block, ss->outer,
            ss->curr, ss->prev_top);
  ICHECK(VALIDSYM(encl) && "Local variable does not have a valid scope");

  /* Now verify that the 'encl' scope is accessible in one of the
     active scopes in the current block.

     It is possible that multiple non-overlapping scopes were pushed
     and popped in the current basic block. When that happens, these
     scopes will appear on the scope stack from ss->outer to
     ss->curr. These scopes are pushed even though they are not
     nested. They will be popped again by verify_endscopes().

     We are satisfied if encl contains any scope from this set. */
  for (i = ss->outer; i <= ss->curr; i++) {
    if (scope_contains(encl, ss->scope[i].block))
      return;
  }

  /* Finally, it is also possible that encl overlaps one of the scopes
     that ended in the current block. ss->prev_top records the
     innermost active scope on entry to the block.  The assertion here
     covers the above scope containment checks too. */
  ICHECK(scope_contains(encl, ss->prev_top) &&
         "Local variable referenced by code outside its scope");
}

/**
   \brief Process all the scope closing labels in bih.

   Pop the closed scopes off the stack.
 */
static void
verify_endscopes(int bih, struct scope_stack *ss)
{
  int ilt, i;
  int ilix;
  int lab, blk;

  for (ilt = BIH_ILTFIRST(bih); ilt; ilt = ILT_NEXT(ilt)) {
    ilix = ILT_ILIP(ilt);
    if (ILI_OPC(ilix) != IL_LABEL)
      continue;
    lab = ILI_OPND(ilix, 1);
    if (!ENDSCOPEG(lab))
      continue;

    /* This is a scope-closing label. */
    blk = ENCLFUNCG(lab);
    verify_block(blk);
    ICHECK(ENDSCOPELABG(blk) == lab);

    /* Find the scope that is being closed. It doesn't have to be the
       topmost one. */
    for (i = ss->curr; i > 0; i--) {
      if (ss->scope[i].block == blk) {
        /* We can't have duplicate labels closing the same scope. */
        ICHECK(ss->scope[i].closing_ilt == 0);
        ss->scope[i].closing_ilt = ilt;
        break;
      }
    }
  }

  /* Pop all the closed scopes. */
  while (ss->scope[ss->curr].closing_ilt != 0)
    pop_block(ss);
}

/**
   \brief Verify all of the locals referenced in the current function
   and check the scope labels.
 */
static void
verify_all_blocks(void)
{
  hashset_t locals = hashset_alloc(hash_functions_direct);
  struct scope_stack ss = {-1, 0, 0, NULL, SPTR_NULL};
  int bih;

  dbgprintf("\nVerifying scopes\n");
  init_scope_stack(&ss, GBL_CURRFUNC);

  /* Visit basic blocks in program order. */
  for (bih = gbl.entbih; bih; bih = BIH_NEXT(bih)) {
    int i, j;

    ss.prev_top = ss.scope[ss.curr].block;
    dbgprintf("block:%d in scope %d (%s)\n", bih, ss.prev_top,
              SYMNAME(ss.prev_top));

    /* First get rid of any open scopes that are closed in this block. */
    verify_endscopes(bih, &ss);

    /* Check labels opening scopes, collect all symbols referenced. */
    hashset_clear(locals);
    verify_beginscopes(bih, locals, &ss);

    /* The scope stack is now as deep as it gets in bih.  Check that
       all referenced locals belong to a pushed scope block. */
    hashset_iterate(locals, verify_local, &ss);

    /* Check labels closing scopes again. This takes care of scopes
       that were opened and closed in the same block. */
    verify_endscopes(bih, &ss);

    /* Check that new open scope are perfectly nested with respect to
       other open scopes. */
    for (i = ss.curr; i > ss.outer; i--) {
      if (ss.scope[i].closing_ilt == 0) {
        for (j = i - 1; j > ss.outer; j--)
          if (ss.scope[j].closing_ilt == 0)
            break;
        ICHECK(ENCLFUNCG(ss.scope[i].block) == ss.scope[j].block);
      }
    }
    ss.outer = ss.curr;
  }

  FREE(ss.scope);
  hashset_free(locals);
}

/**
   \brief Verify the integrity of data structures relating to variable
   scopes in the current function. Terminate compilation with a fatal
   error if anything is inconsistent.
 */
void
scope_verify(void)
{
#if DEBUG
  if (XBIT_USE_SCOPE_LABELS) {
    verify_all_blocks();
  }
#endif
}

/**
   \brief Insert a begin-scope label ILM for the scope blksym which
   much be an ST_BLOCK symbol table entry.

   \return the new ST_LABEL symbol table entry.
 */
int
insert_begin_scope_label(int block_sptr)
{
  int lab;

  ICHECK(VALIDSYM(block_sptr) && STYPEG(block_sptr) == ST_BLOCK);
  ICHECK(BEGINSCOPELABG(block_sptr) == 0 && "Scope already has a begin label.");

  if (!XBIT_USE_SCOPE_LABELS)
    return 0;

  lab = getlab();
  BEGINSCOPEP(lab, 1);
  ENCLFUNCP(lab, block_sptr);
  BEGINSCOPELABP(block_sptr, lab);
  addlabel(lab);

  return lab;
}

/**
   \brief Insert an end-scope label ILM for the scope blksym which
   much be an ST_BLOCK symbol table entry.

   \return the new ST_LABEL symbol table entry.
 */
int
insert_end_scope_label(int block_sptr)
{
  int lab;

  ICHECK(VALIDSYM(block_sptr) && STYPEG(block_sptr) == ST_BLOCK);
  ICHECK(ENDSCOPELABG(block_sptr) == 0 && "Scope already has an end label.");

  if (!XBIT_USE_SCOPE_LABELS)
    return 0;

  lab = getlab();
  ENDSCOPEP(lab, 1);
  ENCLFUNCP(lab, block_sptr);
  ENDSCOPELABP(block_sptr, lab);
  addlabel(lab);

  return lab;
}

/*
 * Scope tracking.
 *
 * During inlining, as we're scanning the current function ILMs top to bottom,
 * call track_scope_label(label_sptr) whenever an IM_LABEL is seen in order to
 * keep track of the current scope.
 *
 * The global variable current_scope points to the currently open scope.
 */
int current_scope = 0;

/**
   \brief Reset scope tracking, start from the outer function scope.
 */
void
track_scope_reset()
{
  current_scope = GBL_CURRFUNC;
  dbgprintf("Scope: Reset to current function %d:%s\n", current_scope,
            SYMNAME(current_scope));
}

/**
   \brief Update curent_scope after scanning over an IM_LABEL ilmx.
   The label sptr is the first and only ILM operand on IM_LABEL.
 */
void
track_scope_label(int label)
{
  assert(STYPEG(label) == ST_LABEL, "track_scope_label: Bogus label", label,
         ERR_Severe);

  /* If scope labels are not being inserted, it is impossible to keep
     track of the current scope. */
  if (!XBIT_USE_SCOPE_LABELS)
    return;

  /* Non-scope labels are simply ignored. */
  if (!BEGINSCOPEG(label) && !ENDSCOPEG(label))
    return;

  /* Labels left over after cancel_lexical_block() are also ignored. */
  if (!ENCLFUNCG(label))
    return;

  if (BEGINSCOPEG(label)) {
    dbgprintf("Scope: Enter scope %d:%s from %d:%s. Parent %d:%s.\n",
              ENCLFUNCG(label), SYMNAME(ENCLFUNCG(label)), current_scope,
              SYMNAME(current_scope), ENCLFUNCG(ENCLFUNCG(label)),
              SYMNAME(ENCLFUNCG(ENCLFUNCG(label))));
    current_scope = ENCLFUNCG(label);
    /* A label can't refer to a function scope. */
    assert(STYPEG(current_scope) == ST_BLOCK,
           "track_scope_label: Scope label does not refer to an ST_BLOCK",
           label, ERR_Severe);
  }

  if (ENDSCOPEG(label)) {
    dbgprintf("Scope: Leave scope %d:%s from %d:%s. Parent %d:%s.\n",
              ENCLFUNCG(label), SYMNAME(ENCLFUNCG(label)), current_scope,
              SYMNAME(current_scope), ENCLFUNCG(ENCLFUNCG(label)),
              SYMNAME(ENCLFUNCG(ENCLFUNCG(label))));
    /* This is the scope that's ending: */
    current_scope = ENCLFUNCG(label);
    assert(STYPEG(current_scope) == ST_BLOCK,
           "track_scope_label: Scope label does not refer to an ST_BLOCK",
           label, ERR_Severe);
    /* This is the active scope after leaving the current scope: */
    current_scope = ENCLFUNCG(current_scope);
  }

  if (!flg.inliner || XBIT(117, 0x10000)) {
    assert(current_scope == GBL_CURRFUNC || STYPEG(current_scope) == ST_BLOCK,
           "track_scope_label: Invalid scope label", label, ERR_Severe);
  }
}

/**
   \brief Find all the scope labels in the current ILM block and pass
   them to track_scope_label. This is used for blocks that don't
   contain any calls.

   This function is looking at numilms ILMs in the ilmb.ilm_base
   array. We don't have any ILM_* macros defined to access that area.

   Compare find_bpar() in inliner.c
 */
void
find_scope_labels(int numilms)
{
  int ilmx, len;

  if (!XBIT_USE_SCOPE_LABELS)
    return;

  for (ilmx = BOS_SIZE; ilmx < numilms; ilmx += len) {
    int opc = ilmb.ilm_base[ilmx];
    if (opc == IM_LABEL) {
      int label = ilmb.ilm_base[ilmx + 1];
      track_scope_label(label);
    }

    len = ilms[opc].oprs + 1; /* length for opcode and fixed operands */
    if (IM_VAR(opc))
      len += ilmb.ilm_base[ilmx + 1]; /* include the variable opnds */
  }
}

/*
 * Inliner support.
 *
 * The functions and global variables below are used by the C and Fortran
 * inliners.
 */

int new_callee_scope = 0;

/* Stack of simultaneously open callee scopes.
 *
 * This stack contains scopes whose begin label has been inserted, and whose
 * end label has yet to be inserted.
 *
 * The stack top is equal to new_callee_scope only after calling
 * begin_inlined_scope().
 */
static int *open_callee_scope = NULL;
static int open_callee_scope_count = 0;
static int open_callee_scope_capacity = 0;

/* Check if new_callee_scope is at the top of the open_callee_scope stack.
 *
 * This is false when new_callee_scope is 0 or when it hasn't been pushed yet
 * by begin_inlined_scope().
 */
#define NEW_CALLEE_SCOPE_IS_STACK_TOP \
  (open_callee_scope_count > 0 &&     \
   new_callee_scope == open_callee_scope[open_callee_scope_count - 1])

/**
   \brief The original function scope of the callee being inlined is
   represented as a new ST_BLOCK scope, stored in new_callee_scope.

   Call create_inlined_scope() to create new_callee_scope as a subscope of the
   currently tracked scope. Then call begin_inlined_scope() and
   end_inlined_scope() to insert the begin/end labels.
 */
void
create_inlined_scope(int callee_sptr)
{
  if (!XBIT_USE_SCOPE_LABELS) {
    /* Without scope labels, we don't know the scope containing the
       call site. Conservatively map scope to the entire function
       scope. */
    new_callee_scope = GBL_CURRFUNC;
    return;
  }

  if (!flg.inliner || XBIT(117, 0x10000)) {
    assert(current_scope, "create_inlined_scope: No current scope?", 0,
           ERR_Severe);
  }

  /* An existing callee scope should have been saved or cancelled. */
  if (new_callee_scope) {
    assert(NEW_CALLEE_SCOPE_IS_STACK_TOP,
           "create_inlined_scope: Previous callee scope wasn't saved",
           new_callee_scope, ERR_Fatal);
  }

  NEWSYM(new_callee_scope);
  STYPEP(new_callee_scope, ST_BLOCK);
  NMPTRP(new_callee_scope, NMPTRG(callee_sptr));
  SCOPEP(new_callee_scope, 1);
  ENCLFUNCP(new_callee_scope, current_scope);

  dbgprintf("Scope: Create %d:%s for inlining %d:%s.\n", new_callee_scope,
            SYMNAME(new_callee_scope), callee_sptr, SYMNAME(callee_sptr));
}

/**
   \brief Immediately before inlining a function, create a new
   ST_BLOCK to represent the callee's function scope in the current
   function.

   Insert an IM_LABEL ILM to open the new scope, update the
   new_callee_scope and current_scope variables to refer to the newly
   created scope.
 */
void
begin_inlined_scope(int func_sptr)
{
  int label;

  if (!XBIT_USE_SCOPE_LABELS)
    return;

  /* Make a new scope unless create_inlined_scope() has already been called.
   * Don't attempt to reuse an open scope that has already been pushed. */
  if (!new_callee_scope || NEW_CALLEE_SCOPE_IS_STACK_TOP)
    create_inlined_scope(func_sptr);

  assert(new_callee_scope && STYPEG(new_callee_scope) == ST_BLOCK,
         "begin_inlined_scope: No inlined scope was created", new_callee_scope,
         ERR_Severe);

  /* Push scope onto the stack of open callee scopes. */
  NEED(open_callee_scope_count + 1, open_callee_scope, int,
       open_callee_scope_capacity, open_callee_scope_capacity + 16);
  open_callee_scope[open_callee_scope_count++] = new_callee_scope;

  /* Create begin label here, end label in end_inlined_scope(). */
  label = insert_begin_scope_label(new_callee_scope);
  dbgprintf("Scope: Label %d:%s created, nest=%d.\n", label, SYMNAME(label),
            open_callee_scope_count);

  track_scope_label(label);
}

/**
   \brief After inlining a function, emit a scope-ending label.
 */
void
end_inlined_scope(void)
{
  int label;

  if (!XBIT_USE_SCOPE_LABELS)
    return;

  assert(new_callee_scope && STYPEG(new_callee_scope) == ST_BLOCK,
         "end_inlined_scope: Not currently in an inlined scope",
         new_callee_scope, ERR_Severe);

  label = insert_end_scope_label(new_callee_scope);
  dbgprintf("Scope: Label %d:%s created, next=%d.\n", label, SYMNAME(label),
            open_callee_scope_count);

  /* Pop this scope off the stack. Make new_callee_scope refer to the
   * previous open callee scope. */
  assert(NEW_CALLEE_SCOPE_IS_STACK_TOP,
         "end_inlined_scope: Scope stack corrupted", new_callee_scope,
         ERR_Fatal);
  if (--open_callee_scope_count)
    new_callee_scope = open_callee_scope[open_callee_scope_count - 1];
  else
    new_callee_scope = 0;

  track_scope_label(label);
}

/**
   \brief End currently open callee scopes until
   open_callee_scope_count is new_open_count.
 */
void
end_inlined_scopes(int new_open_count)
{
  while (open_callee_scope_count > new_open_count) {
    new_callee_scope = open_callee_scope[open_callee_scope_count - 1];
    end_inlined_scope();
  }
}

/**
   \brief If create_inlined_scope() was called, but the inlining was
   abandoned before any labels were inserted, call
   cancel_inlined_scope() to clean up.
 */
void
cancel_inlined_scope(void)
{
  if (!XBIT_USE_SCOPE_LABELS)
    return;

  if (!new_callee_scope)
    return;

  /* If the begin label was already inserted, this scope can't be cancelled. */
  if (NEW_CALLEE_SCOPE_IS_STACK_TOP)
    return;

  /* Go back to the scope we came from before calling
   * begin_inlined_scope(). */
  current_scope = ENCLFUNCG(new_callee_scope);
  dbgprintf("Scope: Reverted to %d:%s scope.\n", current_scope,
            SYMNAME(current_scope));

  /* Restore the old stack top into new_callee_scope. */
  if (open_callee_scope_count)
    new_callee_scope = open_callee_scope[open_callee_scope_count - 1];
  else
    new_callee_scope = 0;
}

void
reset_new_callee_scope()
{
  new_callee_scope = 0;
}

/**
   \brief Remove any scope labels from all blocks.  This is done as a
   workaround until all the phases of the compiler accept (or ignore)
   scope labels.
 */
void
remove_scope_labels(void)
{
  int bihx, iltx, nextiltx;
  for (bihx = gbl.entbih; 1; bihx = BIH_NEXT(bihx)) {
    rdilts(bihx);
    for (iltx = BIH_ILTFIRST(bihx); iltx; iltx = nextiltx) {
      nextiltx = ILT_NEXT(iltx);
      if (is_scope_label_ili(ILT_ILIP(iltx)))
        delilt(iltx);
    }
    wrilts(bihx);
    if (BIH_LAST(bihx))
      break;
  }
} /* remove_scope_labels */
