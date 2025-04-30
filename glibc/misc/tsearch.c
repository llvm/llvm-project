/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Bernd Schmidt <crux@Pool.Informatik.RWTH-Aachen.DE>, 1997.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Tree search for red/black trees.
   The algorithm for adding nodes is taken from one of the many "Algorithms"
   books by Robert Sedgewick, although the implementation differs.
   The algorithm for deleting nodes can probably be found in a book named
   "Introduction to Algorithms" by Cormen/Leiserson/Rivest.  At least that's
   the book that my professor took most algorithms from during the "Data
   Structures" course...

   Totally public domain.  */

/* Red/black trees are binary trees in which the edges are colored either red
   or black.  They have the following properties:
   1. The number of black edges on every path from the root to a leaf is
      constant.
   2. No two red edges are adjacent.
   Therefore there is an upper bound on the length of every path, it's
   O(log n) where n is the number of nodes in the tree.  No path can be longer
   than 1+2*P where P is the length of the shortest path in the tree.
   Useful for the implementation:
   3. If one of the children of a node is NULL, then the other one is red
      (if it exists).

   In the implementation, not the edges are colored, but the nodes.  The color
   interpreted as the color of the edge leading to this node.  The color is
   meaningless for the root node, but we color the root node black for
   convenience.  All added nodes are red initially.

   Adding to a red/black tree is rather easy.  The right place is searched
   with a usual binary tree search.  Additionally, whenever a node N is
   reached that has two red successors, the successors are colored black and
   the node itself colored red.  This moves red edges up the tree where they
   pose less of a problem once we get to really insert the new node.  Changing
   N's color to red may violate rule 2, however, so rotations may become
   necessary to restore the invariants.  Adding a new red leaf may violate
   the same rule, so afterwards an additional check is run and the tree
   possibly rotated.

   Deleting is hairy.  There are mainly two nodes involved: the node to be
   deleted (n1), and another node that is to be unchained from the tree (n2).
   If n1 has a successor (the node with a smallest key that is larger than
   n1), then the successor becomes n2 and its contents are copied into n1,
   otherwise n1 becomes n2.
   Unchaining a node may violate rule 1: if n2 is black, one subtree is
   missing one black edge afterwards.  The algorithm must try to move this
   error upwards towards the root, so that the subtree that does not have
   enough black edges becomes the whole tree.  Once that happens, the error
   has disappeared.  It may not be necessary to go all the way up, since it
   is possible that rotations and recoloring can fix the error before that.

   Although the deletion algorithm must walk upwards through the tree, we
   do not store parent pointers in the nodes.  Instead, delete allocates a
   small array of parent pointers and fills it while descending the tree.
   Since we know that the length of a path is O(log n), where n is the number
   of nodes, this is likely to use less memory.  */

/* Tree rotations look like this:
      A                C
     / \              / \
    B   C            A   G
   / \ / \  -->     / \
   D E F G         B   F
                  / \
                 D   E

   In this case, A has been rotated left.  This preserves the ordering of the
   binary tree.  */

#include <assert.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <search.h>

/* Assume malloc returns naturally aligned (alignof (max_align_t))
   pointers so we can use the low bits to store some extra info.  This
   works for the left/right node pointers since they are not user
   visible and always allocated by malloc.  The user provides the key
   pointer and so that can point anywhere and doesn't have to be
   aligned.  */
#define USE_MALLOC_LOW_BIT 1

#ifndef USE_MALLOC_LOW_BIT
typedef struct node_t
{
  /* Callers expect this to be the first element in the structure - do not
     move!  */
  const void *key;
  struct node_t *left_node;
  struct node_t *right_node;
  unsigned int is_red:1;
} *node;

#define RED(N) (N)->is_red
#define SETRED(N) (N)->is_red = 1
#define SETBLACK(N) (N)->is_red = 0
#define SETNODEPTR(NP,P) (*NP) = (P)
#define LEFT(N) (N)->left_node
#define LEFTPTR(N) (&(N)->left_node)
#define SETLEFT(N,L) (N)->left_node = (L)
#define RIGHT(N) (N)->right_node
#define RIGHTPTR(N) (&(N)->right_node)
#define SETRIGHT(N,R) (N)->right_node = (R)
#define DEREFNODEPTR(NP) (*(NP))

#else /* USE_MALLOC_LOW_BIT */

typedef struct node_t
{
  /* Callers expect this to be the first element in the structure - do not
     move!  */
  const void *key;
  uintptr_t left_node; /* Includes whether the node is red in low-bit. */
  uintptr_t right_node;
} *node;

#define RED(N) (node)((N)->left_node & ((uintptr_t) 0x1))
#define SETRED(N) (N)->left_node |= ((uintptr_t) 0x1)
#define SETBLACK(N) (N)->left_node &= ~((uintptr_t) 0x1)
#define SETNODEPTR(NP,P) (*NP) = (node)((((uintptr_t)(*NP)) \
					 & (uintptr_t) 0x1) | (uintptr_t)(P))
#define LEFT(N) (node)((N)->left_node & ~((uintptr_t) 0x1))
#define LEFTPTR(N) (node *)(&(N)->left_node)
#define SETLEFT(N,L) (N)->left_node = (((N)->left_node & (uintptr_t) 0x1) \
				       | (uintptr_t)(L))
#define RIGHT(N) (node)((N)->right_node)
#define RIGHTPTR(N) (node *)(&(N)->right_node)
#define SETRIGHT(N,R) (N)->right_node = (uintptr_t)(R)
#define DEREFNODEPTR(NP) (node)((uintptr_t)(*(NP)) & ~((uintptr_t) 0x1))

#endif /* USE_MALLOC_LOW_BIT */
typedef const struct node_t *const_node;

#undef DEBUGGING

#ifdef DEBUGGING

/* Routines to check tree invariants.  */

#define CHECK_TREE(a) check_tree(a)

static void
check_tree_recurse (node p, int d_sofar, int d_total)
{
  if (p == NULL)
    {
      assert (d_sofar == d_total);
      return;
    }

  check_tree_recurse (LEFT(p), d_sofar + (LEFT(p) && !RED(LEFT(p))),
		      d_total);
  check_tree_recurse (RIGHT(p), d_sofar + (RIGHT(p) && !RED(RIGHT(p))),
		      d_total);
  if (LEFT(p))
    assert (!(RED(LEFT(p)) && RED(p)));
  if (RIGHT(p))
    assert (!(RED(RIGHT(p)) && RED(p)));
}

static void
check_tree (node root)
{
  int cnt = 0;
  node p;
  if (root == NULL)
    return;
  SETBLACK(root);
  for(p = LEFT(root); p; p = LEFT(p))
    cnt += !RED(p);
  check_tree_recurse (root, 0, cnt);
}

#else

#define CHECK_TREE(a)

#endif

/* Possibly "split" a node with two red successors, and/or fix up two red
   edges in a row.  ROOTP is a pointer to the lowest node we visited, PARENTP
   and GPARENTP pointers to its parent/grandparent.  P_R and GP_R contain the
   comparison values that determined which way was taken in the tree to reach
   ROOTP.  MODE is 1 if we need not do the split, but must check for two red
   edges between GPARENTP and ROOTP.  */
static void
maybe_split_for_insert (node *rootp, node *parentp, node *gparentp,
			int p_r, int gp_r, int mode)
{
  node root = DEREFNODEPTR(rootp);
  node *rp, *lp;
  node rpn, lpn;
  rp = RIGHTPTR(root);
  rpn = RIGHT(root);
  lp = LEFTPTR(root);
  lpn = LEFT(root);

  /* See if we have to split this node (both successors red).  */
  if (mode == 1
      || ((rpn) != NULL && (lpn) != NULL && RED(rpn) && RED(lpn)))
    {
      /* This node becomes red, its successors black.  */
      SETRED(root);
      if (rpn)
	SETBLACK(rpn);
      if (lpn)
	SETBLACK(lpn);

      /* If the parent of this node is also red, we have to do
	 rotations.  */
      if (parentp != NULL && RED(DEREFNODEPTR(parentp)))
	{
	  node gp = DEREFNODEPTR(gparentp);
	  node p = DEREFNODEPTR(parentp);
	  /* There are two main cases:
	     1. The edge types (left or right) of the two red edges differ.
	     2. Both red edges are of the same type.
	     There exist two symmetries of each case, so there is a total of
	     4 cases.  */
	  if ((p_r > 0) != (gp_r > 0))
	    {
	      /* Put the child at the top of the tree, with its parent
		 and grandparent as successors.  */
	      SETRED(p);
	      SETRED(gp);
	      SETBLACK(root);
	      if (p_r < 0)
		{
		  /* Child is left of parent.  */
		  SETLEFT(p,rpn);
		  SETNODEPTR(rp,p);
		  SETRIGHT(gp,lpn);
		  SETNODEPTR(lp,gp);
		}
	      else
		{
		  /* Child is right of parent.  */
		  SETRIGHT(p,lpn);
		  SETNODEPTR(lp,p);
		  SETLEFT(gp,rpn);
		  SETNODEPTR(rp,gp);
		}
	      SETNODEPTR(gparentp,root);
	    }
	  else
	    {
	      SETNODEPTR(gparentp,p);
	      /* Parent becomes the top of the tree, grandparent and
		 child are its successors.  */
	      SETBLACK(p);
	      SETRED(gp);
	      if (p_r < 0)
		{
		  /* Left edges.  */
		  SETLEFT(gp,RIGHT(p));
		  SETRIGHT(p,gp);
		}
	      else
		{
		  /* Right edges.  */
		  SETRIGHT(gp,LEFT(p));
		  SETLEFT(p,gp);
		}
	    }
	}
    }
}

/* Find or insert datum into search tree.
   KEY is the key to be located, ROOTP is the address of tree root,
   COMPAR the ordering function.  */
void *
__tsearch (const void *key, void **vrootp, __compar_fn_t compar)
{
  node q, root;
  node *parentp = NULL, *gparentp = NULL;
  node *rootp = (node *) vrootp;
  node *nextp;
  int r = 0, p_r = 0, gp_r = 0; /* No they might not, Mr Compiler.  */

#ifdef USE_MALLOC_LOW_BIT
  static_assert (alignof (max_align_t) > 1, "malloc must return aligned ptrs");
#endif

  if (rootp == NULL)
    return NULL;

  /* This saves some additional tests below.  */
  root = DEREFNODEPTR(rootp);
  if (root != NULL)
    SETBLACK(root);

  CHECK_TREE (root);

  nextp = rootp;
  while (DEREFNODEPTR(nextp) != NULL)
    {
      root = DEREFNODEPTR(rootp);
      r = (*compar) (key, root->key);
      if (r == 0)
	return root;

      maybe_split_for_insert (rootp, parentp, gparentp, p_r, gp_r, 0);
      /* If that did any rotations, parentp and gparentp are now garbage.
	 That doesn't matter, because the values they contain are never
	 used again in that case.  */

      nextp = r < 0 ? LEFTPTR(root) : RIGHTPTR(root);
      if (DEREFNODEPTR(nextp) == NULL)
	break;

      gparentp = parentp;
      parentp = rootp;
      rootp = nextp;

      gp_r = p_r;
      p_r = r;
    }

  q = (struct node_t *) malloc (sizeof (struct node_t));
  if (q != NULL)
    {
      /* Make sure the malloc implementation returns naturally aligned
	 memory blocks when expected.  Or at least even pointers, so we
	 can use the low bit as red/black flag.  Even though we have a
	 static_assert to make sure alignof (max_align_t) > 1 there could
	 be an interposed malloc implementation that might cause havoc by
	 not obeying the malloc contract.  */
#ifdef USE_MALLOC_LOW_BIT
      assert (((uintptr_t) q & (uintptr_t) 0x1) == 0);
#endif
      SETNODEPTR(nextp,q);		/* link new node to old */
      q->key = key;			/* initialize new node */
      SETRED(q);
      SETLEFT(q,NULL);
      SETRIGHT(q,NULL);

      if (nextp != rootp)
	/* There may be two red edges in a row now, which we must avoid by
	   rotating the tree.  */
	maybe_split_for_insert (nextp, rootp, parentp, r, p_r, 1);
    }

  return q;
}
libc_hidden_def (__tsearch)
weak_alias (__tsearch, tsearch)


/* Find datum in search tree.
   KEY is the key to be located, ROOTP is the address of tree root,
   COMPAR the ordering function.  */
void *
__tfind (const void *key, void *const *vrootp, __compar_fn_t compar)
{
  node root;
  node *rootp = (node *) vrootp;

  if (rootp == NULL)
    return NULL;

  root = DEREFNODEPTR(rootp);
  CHECK_TREE (root);

  while (DEREFNODEPTR(rootp) != NULL)
    {
      root = DEREFNODEPTR(rootp);
      int r;

      r = (*compar) (key, root->key);
      if (r == 0)
	return root;

      rootp = r < 0 ? LEFTPTR(root) : RIGHTPTR(root);
    }
  return NULL;
}
libc_hidden_def (__tfind)
weak_alias (__tfind, tfind)


/* Delete node with given key.
   KEY is the key to be deleted, ROOTP is the address of the root of tree,
   COMPAR the comparison function.  */
void *
__tdelete (const void *key, void **vrootp, __compar_fn_t compar)
{
  node p, q, r, retval;
  int cmp;
  node *rootp = (node *) vrootp;
  node root, unchained;
  /* Stack of nodes so we remember the parents without recursion.  It's
     _very_ unlikely that there are paths longer than 40 nodes.  The tree
     would need to have around 250.000 nodes.  */
  int stacksize = 40;
  int sp = 0;
  node **nodestack = alloca (sizeof (node *) * stacksize);

  if (rootp == NULL)
    return NULL;
  p = DEREFNODEPTR(rootp);
  if (p == NULL)
    return NULL;

  CHECK_TREE (p);

  root = DEREFNODEPTR(rootp);
  while ((cmp = (*compar) (key, root->key)) != 0)
    {
      if (sp == stacksize)
	{
	  node **newstack;
	  stacksize += 20;
	  newstack = alloca (sizeof (node *) * stacksize);
	  nodestack = memcpy (newstack, nodestack, sp * sizeof (node *));
	}

      nodestack[sp++] = rootp;
      p = DEREFNODEPTR(rootp);
      if (cmp < 0)
	{
	  rootp = LEFTPTR(p);
	  root = LEFT(p);
	}
      else
	{
	  rootp = RIGHTPTR(p);
	  root = RIGHT(p);
	}
      if (root == NULL)
	return NULL;
    }

  /* This is bogus if the node to be deleted is the root... this routine
     really should return an integer with 0 for success, -1 for failure
     and errno = ESRCH or something.  */
  retval = p;

  /* We don't unchain the node we want to delete. Instead, we overwrite
     it with its successor and unchain the successor.  If there is no
     successor, we really unchain the node to be deleted.  */

  root = DEREFNODEPTR(rootp);

  r = RIGHT(root);
  q = LEFT(root);

  if (q == NULL || r == NULL)
    unchained = root;
  else
    {
      node *parentp = rootp, *up = RIGHTPTR(root);
      node upn;
      for (;;)
	{
	  if (sp == stacksize)
	    {
	      node **newstack;
	      stacksize += 20;
	      newstack = alloca (sizeof (node *) * stacksize);
	      nodestack = memcpy (newstack, nodestack, sp * sizeof (node *));
	    }
	  nodestack[sp++] = parentp;
	  parentp = up;
	  upn = DEREFNODEPTR(up);
	  if (LEFT(upn) == NULL)
	    break;
	  up = LEFTPTR(upn);
	}
      unchained = DEREFNODEPTR(up);
    }

  /* We know that either the left or right successor of UNCHAINED is NULL.
     R becomes the other one, it is chained into the parent of UNCHAINED.  */
  r = LEFT(unchained);
  if (r == NULL)
    r = RIGHT(unchained);
  if (sp == 0)
    SETNODEPTR(rootp,r);
  else
    {
      q = DEREFNODEPTR(nodestack[sp-1]);
      if (unchained == RIGHT(q))
	SETRIGHT(q,r);
      else
	SETLEFT(q,r);
    }

  if (unchained != root)
    root->key = unchained->key;
  if (!RED(unchained))
    {
      /* Now we lost a black edge, which means that the number of black
	 edges on every path is no longer constant.  We must balance the
	 tree.  */
      /* NODESTACK now contains all parents of R.  R is likely to be NULL
	 in the first iteration.  */
      /* NULL nodes are considered black throughout - this is necessary for
	 correctness.  */
      while (sp > 0 && (r == NULL || !RED(r)))
	{
	  node *pp = nodestack[sp - 1];
	  p = DEREFNODEPTR(pp);
	  /* Two symmetric cases.  */
	  if (r == LEFT(p))
	    {
	      /* Q is R's brother, P is R's parent.  The subtree with root
		 R has one black edge less than the subtree with root Q.  */
	      q = RIGHT(p);
	      if (RED(q))
		{
		  /* If Q is red, we know that P is black. We rotate P left
		     so that Q becomes the top node in the tree, with P below
		     it.  P is colored red, Q is colored black.
		     This action does not change the black edge count for any
		     leaf in the tree, but we will be able to recognize one
		     of the following situations, which all require that Q
		     is black.  */
		  SETBLACK(q);
		  SETRED(p);
		  /* Left rotate p.  */
		  SETRIGHT(p,LEFT(q));
		  SETLEFT(q,p);
		  SETNODEPTR(pp,q);
		  /* Make sure pp is right if the case below tries to use
		     it.  */
		  nodestack[sp++] = pp = LEFTPTR(q);
		  q = RIGHT(p);
		}
	      /* We know that Q can't be NULL here.  We also know that Q is
		 black.  */
	      if ((LEFT(q) == NULL || !RED(LEFT(q)))
		  && (RIGHT(q) == NULL || !RED(RIGHT(q))))
		{
		  /* Q has two black successors.  We can simply color Q red.
		     The whole subtree with root P is now missing one black
		     edge.  Note that this action can temporarily make the
		     tree invalid (if P is red).  But we will exit the loop
		     in that case and set P black, which both makes the tree
		     valid and also makes the black edge count come out
		     right.  If P is black, we are at least one step closer
		     to the root and we'll try again the next iteration.  */
		  SETRED(q);
		  r = p;
		}
	      else
		{
		  /* Q is black, one of Q's successors is red.  We can
		     repair the tree with one operation and will exit the
		     loop afterwards.  */
		  if (RIGHT(q) == NULL || !RED(RIGHT(q)))
		    {
		      /* The left one is red.  We perform the same action as
			 in maybe_split_for_insert where two red edges are
			 adjacent but point in different directions:
			 Q's left successor (let's call it Q2) becomes the
			 top of the subtree we are looking at, its parent (Q)
			 and grandparent (P) become its successors. The former
			 successors of Q2 are placed below P and Q.
			 P becomes black, and Q2 gets the color that P had.
			 This changes the black edge count only for node R and
			 its successors.  */
		      node q2 = LEFT(q);
		      if (RED(p))
			SETRED(q2);
		      else
			SETBLACK(q2);
		      SETRIGHT(p,LEFT(q2));
		      SETLEFT(q,RIGHT(q2));
		      SETRIGHT(q2,q);
		      SETLEFT(q2,p);
		      SETNODEPTR(pp,q2);
		      SETBLACK(p);
		    }
		  else
		    {
		      /* It's the right one.  Rotate P left. P becomes black,
			 and Q gets the color that P had.  Q's right successor
			 also becomes black.  This changes the black edge
			 count only for node R and its successors.  */
		      if (RED(p))
			SETRED(q);
		      else
			SETBLACK(q);
		      SETBLACK(p);

		      SETBLACK(RIGHT(q));

		      /* left rotate p */
		      SETRIGHT(p,LEFT(q));
		      SETLEFT(q,p);
		      SETNODEPTR(pp,q);
		    }

		  /* We're done.  */
		  sp = 1;
		  r = NULL;
		}
	    }
	  else
	    {
	      /* Comments: see above.  */
	      q = LEFT(p);
	      if (RED(q))
		{
		  SETBLACK(q);
		  SETRED(p);
		  SETLEFT(p,RIGHT(q));
		  SETRIGHT(q,p);
		  SETNODEPTR(pp,q);
		  nodestack[sp++] = pp = RIGHTPTR(q);
		  q = LEFT(p);
		}
	      if ((RIGHT(q) == NULL || !RED(RIGHT(q)))
		  && (LEFT(q) == NULL || !RED(LEFT(q))))
		{
		  SETRED(q);
		  r = p;
		}
	      else
		{
		  if (LEFT(q) == NULL || !RED(LEFT(q)))
		    {
		      node q2 = RIGHT(q);
		      if (RED(p))
			SETRED(q2);
		      else
			SETBLACK(q2);
		      SETLEFT(p,RIGHT(q2));
		      SETRIGHT(q,LEFT(q2));
		      SETLEFT(q2,q);
		      SETRIGHT(q2,p);
		      SETNODEPTR(pp,q2);
		      SETBLACK(p);
		    }
		  else
		    {
		      if (RED(p))
			SETRED(q);
		      else
			SETBLACK(q);
		      SETBLACK(p);
		      SETBLACK(LEFT(q));
		      SETLEFT(p,RIGHT(q));
		      SETRIGHT(q,p);
		      SETNODEPTR(pp,q);
		    }
		  sp = 1;
		  r = NULL;
		}
	    }
	  --sp;
	}
      if (r != NULL)
	SETBLACK(r);
    }

  free (unchained);
  return retval;
}
libc_hidden_def (__tdelete)
weak_alias (__tdelete, tdelete)


/* Walk the nodes of a tree.
   ROOT is the root of the tree to be walked, ACTION the function to be
   called at each node.  LEVEL is the level of ROOT in the whole tree.  */
static void
trecurse (const void *vroot, __action_fn_t action, int level)
{
  const_node root = (const_node) vroot;

  if (LEFT(root) == NULL && RIGHT(root) == NULL)
    (*action) (root, leaf, level);
  else
    {
      (*action) (root, preorder, level);
      if (LEFT(root) != NULL)
	trecurse (LEFT(root), action, level + 1);
      (*action) (root, postorder, level);
      if (RIGHT(root) != NULL)
	trecurse (RIGHT(root), action, level + 1);
      (*action) (root, endorder, level);
    }
}


/* Walk the nodes of a tree.
   ROOT is the root of the tree to be walked, ACTION the function to be
   called at each node.  */
void
__twalk (const void *vroot, __action_fn_t action)
{
  const_node root = (const_node) vroot;

  CHECK_TREE ((node) root);

  if (root != NULL && action != NULL)
    trecurse (root, action, 0);
}
libc_hidden_def (__twalk)
weak_alias (__twalk, twalk)

/* twalk_r is the same as twalk, but with a closure parameter instead
   of the level.  */
static void
trecurse_r (const void *vroot, void (*action) (const void *, VISIT, void *),
	    void *closure)
{
  const_node root = (const_node) vroot;

  if (LEFT(root) == NULL && RIGHT(root) == NULL)
    (*action) (root, leaf, closure);
  else
    {
      (*action) (root, preorder, closure);
      if (LEFT(root) != NULL)
	trecurse_r (LEFT(root), action, closure);
      (*action) (root, postorder, closure);
      if (RIGHT(root) != NULL)
	trecurse_r (RIGHT(root), action, closure);
      (*action) (root, endorder, closure);
    }
}

void
__twalk_r (const void *vroot, void (*action) (const void *, VISIT, void *),
	   void *closure)
{
  const_node root = (const_node) vroot;

  CHECK_TREE ((node) root);

  if (root != NULL && action != NULL)
    trecurse_r (root, action, closure);
}
libc_hidden_def (__twalk_r)
weak_alias (__twalk_r, twalk_r)

/* The standardized functions miss an important functionality: the
   tree cannot be removed easily.  We provide a function to do this.  */
static void
tdestroy_recurse (node root, __free_fn_t freefct)
{
  if (LEFT(root) != NULL)
    tdestroy_recurse (LEFT(root), freefct);
  if (RIGHT(root) != NULL)
    tdestroy_recurse (RIGHT(root), freefct);
  (*freefct) ((void *) root->key);
  /* Free the node itself.  */
  free (root);
}

void
__tdestroy (void *vroot, __free_fn_t freefct)
{
  node root = (node) vroot;

  CHECK_TREE (root);

  if (root != NULL)
    tdestroy_recurse (root, freefct);
}
libc_hidden_def (__tdestroy)
weak_alias (__tdestroy, tdestroy)
