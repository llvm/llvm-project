/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ILI_REWRITE_H_
#define ILI_REWRITE_H_

/*
 * A general postorder (bottom-up) ILI rewriting callback framework.
 *
 * Every statement (ILT) in the program is visited once in some arbitrary
 * block order.  For each statement, the ILIs are visited in postorder
 * (leaves before interior nodes).  Each ILI, as it is visited, is passed
 * to a callback, which can either return it unchanged or replace it
 * with a new ILI.
 *
 * If the callback replaces an ILI, the outer expression that linked to it
 * will have that ILI link operand provisionally replaced before it is
 * passed to the callback in its turn.  In other words, the callback routine
 * may safely assume that every ILI that is presented to the callback
 * already reflects any changes that it may have made to its subexpressions.
 *
 * This rewriting framework automatically detects when this rewriting
 * of operands prior to the invocation of the callback leads to improvement
 * of the program.  After the operands of an ILI are processed, if any
 * have changed, the ILI is reconstructed by calling addili(), and the
 * index that addili() returns is examined to see whether addili() itself
 * rewrote the expression.
 *
 * The callback is supplied with a pointer to a context description
 * structure that includes the ILI index, its statement context, its original
 * form, and other information.
 *
 * Be advised: this framework knows about the subtle semantics of JSR
 * and CSE operations, and will "do the right thing" to their operands
 * in order to propagate any other ILI replacements returned by the
 * visitation callback.  However, the framework is unable to prevent
 * the callback from messing up the CSE semantics.
 *
 * Be advised: the framework does not automatically adjust reference
 * counts on labels.  It originall did so, but it proved to be impossible
 * to distinguish iliutil.c's changes from callback changes, and
 * iliutil manages RFCNT fields explicitly.
 */

#include "fastset.h"

struct ILI_coordinates;

/* Callback function type for ILI rewriting.
 * Declare callbacks for visit_ilis() like this:
 *
 *   int my_callback(const ILI_coordinates *at) {
 *	struct my_context *context = at->context;
 *	if (my_transformation)
 *	    return new_ili_index;
 *	return at->ili; // no change
 *   }
 */
typedef int (*ILI_visitor)(const struct ILI_coordinates *at);

typedef bool (*ILI_tree_scan_visitor)(void *visitor_context, int ili);

/**
   \brief Visitation context structure passed to callbacks.
 */
typedef struct ILI_coordinates {
  ILI_visitor visitor;
  void *context;    /**< callback's own state */
  int original_ili; /**< preorder value, before recursion into operands */
  int ili;     /**< after recursion into operands (might be != original_ili) */
  int bih;
  int ilt;     /**< position of statement containing this ILI instance */
  /** null if ILI is root expression of ILT */
  const struct ILI_coordinates *parent;
  int parent_opnd;  /**< ILI_OPND(parent->ili,parent_opnd) links here */
  /** this ILI improved when operands updated */
  bool this_ili_improved;
  bool has_cse;     /**< operand trees contain JSR and/or CSE */
} ILI_coordinates;

#ifdef __cplusplus
inline fastset *GetLiveILIs(const ILI_coordinates *coor) {
  return static_cast<fastset*>(coor->context);
}
#else
#define GetLiveILIs(coor)  (coor->context)
#endif

/**
   \brief ILI rewriting driver.  Returns TRUE if any change occurs.

   The traversal is in block sequence order, with the ILTs being visited in
   forward order within each block.

   The void *visitor_context is passed to the visitation callback as
   at->context.

   Set the "context_insensitive" flag to TRUE for faster processing when the
   rewriting visitor is known to not be sensitive to the context (i.e., it
   always maps ILI 'x' to the same 'y' in any statement).
 */
bool visit_ilis(ILI_visitor visitor, void *visitor_context,
                bool context_insensitive);

/**
   \brief Utility for callbacks: given an ILI index, create a new ILI in which
   one operand has been replaced.
 */
int update_ili_operand(int ili, int opnd_index, int new_opnd);

/**
   \brief Collects the set of ILT root expression ILIs.
 */
void collect_root_ilis(fastset *root_ilis);

/**
   \brief Collects the set of live ILIs.
 */
void collect_live_ilis(fastset *live);

/**
   \brief Collects the set of ILIs in an ILI expression tree.
 */
void collect_tree_ilis(fastset *tree_ilis, int ili, bool scan_cses);

/**
   \brief ILI tree scanning in preorder with a callback.
   A true return from the callback causes the scan to immediately return TRUE.
   Does not descend into IL_CSExx operands; recurse thither from the visitor if
   that's what you really want.
 */
bool scan_ili_tree(ILI_tree_scan_visitor visitor, void *visitor_context,
                   int ili);

#endif /* ILI_REWRITE_H_ */
