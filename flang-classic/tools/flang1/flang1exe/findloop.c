/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * \brief optimizer submodule to find loops in a flow graph. Used by the
 * optimizer and vectorizer.
 */

#include "findloop.h"
#include "gbldefs.h"
#ifndef FE90
#include "fgraph.h"
#include "ili.h"
#include "regutil.h"
#include "machreg.h"
#else
#include "error.h"
#include "global.h"
#include "symtab.h"
#include "ast.h"
#include "nme.h"
#endif

#include "optimize.h"
#ifndef FE90
#include "apt.h"
#endif

static void top_sort(void);
static void build_loop(int);
static void add_to_region(int);
static void add_lpexit(int, int);
static void unvisit(int, int);
static void malformed(int, int);
static void add_to_malf(int);
static void convert_loop(int);

static int current_lp;
static int current_lp_tail;
static int lp_topsort;
static int naturalloop;

/*********************************************************************/

/** \brief A routine to find the loops in a flow graph
 */
void
findloop(int hlopt_bv)
{
  int edge, lp, head, tail, max_level, i;
  LP *p;
  int exit;
  int pt;
  bool precedes;
  PSI_P q;
#ifndef FE90
  int save_rgset0, *save_rtemps0;
#endif
  bool any_malformed;

  if (OPTDBG(9, 8))
    fprintf(gbl.dbgfil, "\n---------- findloop trace for function \"%s\"\n",
            getprint(BIH_LABEL(gbl.entbih)));

#if DEBUG
  if (XBIT(6, 0x100000))
    lp_topsort = 1;
#endif

  opt.nloops = 0;
  opt.lpb.stg_avail = 1;
/* Optimizer sets two fields in opt.lpb.stg_avail[0] before findloop is called.
 * Save and restore those */
#ifndef FE90
  save_rgset0 = LP_RGSET(0);
  save_rtemps0 = LP_RTEMPS(0);
#endif
  STG_CLEAR(opt.lpb, 0);
#ifndef FE90
  LP_RGSET(0) = save_rgset0;
  LP_RTEMPS(0) = save_rtemps0;
  LP_HEAD(0) = BIH_TO_FG(gbl.entbih);
  LP_TAIL(0) = BIH_TO_FG(opt.exitbih);
#else
  LP_HEAD(0) = BIH_TO_FG(gbl.entbih);
  LP_TAIL(0) = opt.exitfg;
#endif
  LP_CNCALL(0) = 1;
  LP_TAIL_AEXE(0) = 1;

  if (NUM_RTE == 0)
    goto build_region0;
  /*
   * go through the retreating edges and find the back edges using the
   * dominance requirement.  Note that a non-back edge is denoted by
   * by a 0 successor field.
   */
  for (edge = 0; edge < NUM_RTE; edge++) {
    if (!is_dominator((int)EDGE_SUCC(edge), (int)EDGE_PRED(edge))) {
      if (OPTDBG(9, 8))
        fprintf(gbl.dbgfil, "edge %d (%d, %d) is not back edge\n", edge,
                EDGE_PRED(edge), EDGE_SUCC(edge));
      EDGE_SUCC(edge) = 0;
    }
  }
  /*
   * Now, those back edges which have identical heads are linked together
   * (i.e., the natural loops of the edges with the same head are combined
   * into a single loop).  Look ahead in the table of edges.  If an edge
   * with the same "head" is found, link the current edge to it.  Also,
   * the "tail" with the maximum dfn is computed which will become
   * the tail of the loop; this ensures that if "head==tail", we have a
   * one block loop.  The list will be used to find the other edges
   * (actually the extra tails) so that the build of the loop is done
   * correctly.
   *
   * NOTE that a back edge which is added to the list will have its
   * successor field set to zero.
   */
  for (edge = 0; edge < NUM_RTE; edge++) {
    head = EDGE_SUCC(edge);
    if (head == 0)
      continue;             /* not a back edge or already processed */
    tail = EDGE_PRED(edge); /* tail with maximum dfn */
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil, "back edge %d, (%d %d)\n", edge, tail, head);
    for (i = edge + 1; i < NUM_RTE; i++) {
      /*
       * as edges with the same head are found, ensure that the
       * edges' tails lexically follow their respective heads.
       * NOTE that this only checks the multiple edges which follow
       * this edge. Single edges and the first edge of the multiple
       * case are checked in the final scan of the edges.
       */
      precedes = false;
      if (head == EDGE_SUCC(i)) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil, "    add tail %d to edge\n", EDGE_PRED(i));
        if (EDGE_PRED(i) < EDGE_SUCC(i)) {
          if (OPTDBG(9, 8))
            fprintf(gbl.dbgfil,
                    "    mult.edge %d (%d %d), tail precedes head\n", i,
                    EDGE_PRED(i), EDGE_SUCC(i));
          precedes = true;
        }
        EDGE_NEXT(i) = EDGE_NEXT(edge);
        EDGE_NEXT(edge) = i;
        if (FG_DFN(tail) < FG_DFN(EDGE_PRED(i))) {
          if (OPTDBG(9, 8))
            fprintf(gbl.dbgfil, "    tail exchange: old %d, new %d\n", tail,
                    EDGE_PRED(i));
          /*  exchange predecessors (tails) of the two edges */
          tail = EDGE_PRED(i); /* new maximum */
          EDGE_PRED(i) = EDGE_PRED(edge);
          EDGE_PRED(edge) = tail;
        }
        EDGE_SUCC(i) = 0;
      }
      if (precedes) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil,
                  "    multiple edge %d (%d %d), tail precedes head\n", edge,
                  tail, EDGE_SUCC(edge));
        EDGE_SUCC(edge) = 0;
      }
    }
  }
  /*
   * Go through the back edges and create an entry in the loop table;
   * don't allow those whose tail lexically precedes the head.
   */
  any_malformed = false;
  max_level = 0;
  for (edge = 0; edge < NUM_RTE; edge++) {
    head = EDGE_SUCC(edge);
    if (head == 0)
      continue;
    tail = EDGE_PRED(edge);
    if (head <= tail) {
      opt.nloops = STG_NEXT(opt.lpb);
      p = opt.lpb.stg_base + opt.nloops;
      BZERO(p, LP, 1);
      p->head = head;
      p->tail = tail;
      p->level = 1;
      p->exits = PSI_P_NULL;
      p->flags.bits.innermost = 1; /* loop is innermost unless
                                    * proven otherwise  */
      p->flags.bits.cncall = 1;    /* loop is cncall unless
                                    * proven otherwise */
      p->edge = edge;

#ifndef FE90
      /*
       * allocate an area for the loop's register used information and
       * the rtemps information - zero them out.
       */
      p->rused = (RUSED *)getitem(RUSED_AREA, MR_NUMGLB * sizeof(RUSED));
      BZERO(p->rused, RUSED, MR_NUMGLB);
      p->rtemps = (int *)getitem(RUSED_AREA, RTEMPS * sizeof(int));
      BZERO(p->rtemps, int, RTEMPS);
#endif
#if DEBUG
      if (OPTDBG(9, 8)) {
        assert(tail == head || !is_dominator(tail, head), "bad dominator rel",
               head, ERR_Informational);
        fprintf(gbl.dbgfil, "---LOOPS--- %d is (%d, %d)\n", opt.nloops, tail,
                head);
      }
#endif
    } else {
      if (OPTDBG(9, 8))
        fprintf(gbl.dbgfil, "edge %d (%d %d), tail precedes.0 head\n", edge,
                tail, head);
      any_malformed = true;
    }
    /*next_edge: ;*/
  }

  /*
   * go through the loops and determine their level and parent values
   */
  for (lp = 1; lp <= opt.nloops; lp++) {
    p = opt.lpb.stg_base + lp;
    p->exits = PSI_P_NULL;
    /*
     * if this a 1 block loop, there is no need to determine if any loops
     * are contained in it
     */
    if ((p->head) == (p->tail)) {
      for (q = FG_SUCC(p->tail); q != PSI_P_NULL; q = PSI_NEXT(q))
        if ((exit = PSI_NODE(q)) != p->tail) {
          add_lpexit(lp, exit);
          PSI_EXIT(q) = 1;
        }
      continue;
    }
    /*
     * otherwise, collect all the blocks which are in the loop -- this
     * "region" will contain all of the blocks in loops which are
     * contained in the loop
     */
    build_loop(lp);
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil, "for natural loop %5d\n", lp);

    /*  scan natural loop to find if there are multiple exits and
     *  detect any extended range loops.
     */

    for (i = LP_FG(lp); i != 0; i = FG_NEXT(i)) {
      if (OPTDBG(9, 8))
        fprintf(gbl.dbgfil, " - fg %5d  bih %5d\n", i, FG_TO_BIH(i));
      for (q = FG_SUCC(i); q != PSI_P_NULL; q = PSI_NEXT(q)) {
        if (FG_LOOP(exit = PSI_NODE(q)) != lp) {
          add_lpexit(lp, exit);
          PSI_EXIT(q) = 1;
        }
      }
      if (i < LP_HEAD(lp) || i > LP_TAIL(lp)) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil, " - fg %d is not enclosed within loop\n", i);
        /*
        fprintf(stderr,
            "loop %d, edge (%d %d) does not enclose node %d\n",
            lp, LP_TAIL(lp), LP_HEAD(lp), i);
        */
        LP_XTNDRNG(lp) = 1;
      }
    }
    /*
     * scan through the other loops to determine if they are contained in
     * the loop lp.  This is determined by checking the loop field for
     * the flow graph nodes of the head and tail of their back edges
     */
    for (i = 1; i <= opt.nloops; i++) {
      if (i == lp)
        continue;

      /*
       * if both the head and the tail of the back edge are in the
       * loop, then this loop is contained in lp
       */
      if (FG_LOOP(LP_HEAD(i)) == 0)
        continue;
      if (FG_LOOP(LP_TAIL(i)) == 0)
        continue;

      /*
       * increment the level of the contained loop.  Also, set the
       * maximum level if necessary
       */
      if (max_level < (++LP_LEVEL(i)))
        max_level = LP_LEVEL(i);

      /*
       * lp is the parent of loop i if this is the first time or the
       * level of lp is greater than the level of the current parent of
       * loop i
       */
      if (LP_PARENT(i) == 0 || p->level > LP_LEVEL(LP_PARENT(i))) {
        LP_PARENT(i) = lp;
        LP_INNERMOST(lp) = 0;
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil, "            new parent of %d is %d\n", i, lp);
      }
    }

    /* cleanup the region built by build_loop  */
    /* mark nodes as unvisited and clear FG_LOOP field */
    unvisit(naturalloop, 1);
    LP_FG(lp) = 0;
  }

  /* sort the loops in innermost order  */

  if (opt.nloops)
    top_sort();

  /*
   * go through the loops according to their sorted order to build the
   * regions for each loop.  At this time, a loop's region will exclude any
   * blocks which belong to any enclosed loops
   */
  for (i = 1; i <= opt.nloops; ++i) {
    lp = LP_LOOP(i);
    /*
     * add this loop to its parent's sibling list. must do this for
     * all of the natural loops which were discovered including those
     * which are ignored or merged with other loops (i.e., their
     * FG_LOOP fields are zero).
     */
    pt = LP_PARENT(lp);
    LP_SIBLING(lp) = LP_CHILD(pt);
    LP_CHILD(pt) = lp;
    /* update the count of the loop; its count may be nonzero since
     * all of the loops contained by this loop have been counted.
     */
    LP_COUNT(lp)++;
    /* propagate the loop's count to its parent
     */
    LP_COUNT(pt) += LP_COUNT(lp);

    /*
     * only build the region if the head and tail nodes are not members
     * of another loop
     */
    if (FG_LOOP(LP_HEAD(lp)) || FG_LOOP(LP_TAIL(lp)))
      continue;
    build_loop(lp);

#if DEBUG
    if (OPTDBG(9, 8)) {
      int z;
      fprintf(gbl.dbgfil, "regions for loop %5d\n", lp);
      for (z = LP_FG(lp); z != 0; z = FG_NEXT(z))
        fprintf(gbl.dbgfil, " - fg %5d  bih %5d\n", z, FG_TO_BIH(z));
    }
#endif
    /* mark nodes as unvisited , but don't clear FG_LOOP field */
    unvisit(naturalloop, 0);

    /*
     * if lp is some type of while loop, convert it to an "if - repeat"
     * loop
     */

    if (hlopt_bv & HLOPT_ENDTEST)
      convert_loop(lp);

    LP_TAIL_AEXE(lp) = is_tail_aexe(lp);
  }
  if (any_malformed) {
    for (edge = 0; edge < NUM_RTE; edge++) {
      head = EDGE_SUCC(edge);
      if (head == 0)
        continue;
      tail = EDGE_PRED(edge);
      if (head > tail) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil, "edge %d (%d %d), tail precedes.1 head\n", edge,
                  tail, head);
        malformed(head, tail);
      }
    }
  }

/* build region 0  */

build_region0:
  p = opt.lpb.stg_base;
  p->fg = 0;
  p->flags.bits.callfg = BIH_EX(gbl.entbih);
  for (i = opt.dfn; i; i--) {
    head = VTX_NODE(i);
    if (FG_LOOP(head) == 0) {
      FG_NEXT(head) = p->fg;
      p->fg = head;
    }
  }
#ifndef FE90
  LP_RUSED(0) = (RUSED *)getitem(RUSED_AREA, MR_NUMGLB * sizeof(RUSED));
  BZERO(LP_RUSED(0), RUSED, MR_NUMGLB);
  /*
   * NOTE:  LP_RTEMPS(0) has already been allocated by function_init
   *        in optimize.c
   */

  /*
   * Now that loops have  been found, we can compute post_dominance
   * and build the control dependence data structures. Necessary
   * to find loops first since we identify infinite loops to
   * insert pseudo edges to the exit node. We need a path from exit
   * to every node in order to compute the post_dominator tree.
   */
  build_apt(0); /* 1 = dominance frontiers, 0 = control dependence */
#endif
}

/** \brief findlooptopsort builds loops and adds the FG nodes to the loop in
 * topological order, top-down.
 */
void
findlooptopsort(void)
{
  int savetopsort;
  savetopsort = lp_topsort;
  lp_topsort = 1;
  findloop(HLOPT_ALL);
  lp_topsort = savetopsort;
} /* findlooptopsort */

/*********************************************************************/

/** \brief Topological sort
 *
 * This routine is used to sort the loops found in the flowgraph such that a
 * loop X is processed for optimizations before any of the loops containing X.
 * The algorithm used is the topological sort as defined in Knuth (p. 262), The
 * Art of Computer Programming, Volume 1/Fundamental Algorithms.
 */
static void
top_sort(void)
{

#define TOP(i) topb[i].top
#define COUNT(i) topb[i].count
#define QLINK(i) topb[i].count
#define SUCC(p) p->succ
#define NEXT(p) p->next

  typedef struct topi_tag {
    short succ;
    struct topi_tag *next;
  } TOPI;
  typedef struct {
    short count;
    TOPI *top;
  } TOPB;
  TOPB *topb;
  TOPI *p;
  int k, r;
  int lp;
  int n;

  /*
   * allocate storage for the top array and intitalize the count and top
   * fields.  There is a need for a top entry for each loop and loop "0",
   * and one for linking together the entries whose counts are zero. Entry
   * 0 is reserved for the "level-0" loop which represents the outermost
   * region of a function and contains all loops. Entries 1 through nloops
   * are used for the respective loops.  Entry n is only used for linking
   * up those loops which have zero counts (i.e, the innermost loops).
   */
  n = opt.nloops + 1;
  NEW(topb, TOPB, n + 1);
  for (k = 0; k < n; k++) {
    TOP(k) = NULL;
    COUNT(k) = 0;
  }

  /*
   * go through the loops and define the partial ordering, lp < k, where k
   * is the parent of lp (i.e., lp is contained in k). Note that at least
   * one relation of the form i < 0 exists.
   */
  for (lp = 1; lp <= opt.nloops; lp++) {
    k = LP_PARENT(lp); /* relation lp < k */
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil, "            relation %d < %d\n", lp, k);
    COUNT(k)++; /* # predecessors of k */
    p = (TOPI *)getitem(TOPI_AREA, sizeof(TOPI));
    SUCC(p) = k; /* k is a successor of lp */
    NEXT(p) = TOP(lp);
    TOP(lp) = p;
  }

  /*
   * Initialize the queue used to hold loops in topological order by
   * scanning for loops which have no predecessors (loop 0 has at least one
   * predecessor.  The list produced is always non- empty given that there
   * are loops
   */

  r = n;
  for (k = 1; k < n; k++)
    if (COUNT(k) == 0) {
      r = QLINK(r) = k;
      if (OPTDBG(9, 8))
        fprintf(gbl.dbgfil, "            innermost loop %d\n", k);
    }
  assert(r != n, "top_sort: wrong qlink", r, ERR_Severe);

  /*
   * Go through the relations and create the order - this continues until
   * loop zero is seen (it is always the last one seen)
   */

  k = 0;
  for (lp = QLINK(n); lp > 0; lp = QLINK(lp)) {
    LP_LOOP(++k) = lp;
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil, "            next loop: %d\n", lp);
    n--;
    for (p = TOP(lp); p != NULL; p = NEXT(p))
      if (--COUNT(SUCC(p)) == 0)
        r = QLINK(r) = SUCC(p);
  }
  assert(n == 1, "wrong top_sort", n, ERR_Severe);

  /* free up the top array and the area used for the successors  */

  FREE(topb);
  freearea(TOPI_AREA);
}

/*********************************************************************/

/** \brief Determine if a node dominates another
 *
 * \param v first node
 * \param w second node
 * \return true if 'v' dominates 'w'
 *
 * Walk up dominator tree from 'w', stop at 'v' (return true) or when we reach
 * a node above 'v' in the spanning tree.
 */
bool
is_dominator(int v, int w)
{
  int vv, vw;
  vv = FG_DFN(v);
  if (v == 0)
    return true;
  while (w != 0) {
    if (v == w)
      return true;
    /* this test is so complicated because either v or w
     * may have been added after the depth-first tree was built
     * and nodes were numbered; if they are both numbered, then
     * we can stop when we reach a 'w' above 'v' in the spanning tree. */
    if (vv > 0 && (vw = FG_DFN(w)) > 0 && vw < vv)
      return false;
    w = FG_DOM(w);
  }
  return false;
} /* is_dominator */

#if defined(FG_PDOM)
/*
 * return true if 'v' postdominates 'w'
 *  walk up postdominator tree from 'w'
 */
bool
is_post_dominator(int v, int w)
{
  int vv;
  for (vv = w; vv > 0; vv = FG_PDOM(vv)) {
    if (vv == v)
      return true;
  }
  return false;
} /* is_post_dominator */
#endif

/*********************************************************************/

/*
 * Determine if the tail of a loop always executed:
 */
bool
is_tail_aexe(int lp)
{
  int fg;
  int cnt;
  PSI_P p;
  if (LP_MEXITS(lp))
    return false;
  fg = LP_TAIL(lp);
  if (LP_HEAD(lp) == fg)
    return true;
  if (!BIH_FT(FG_TO_BIH(fg)))
    /*
     * the multi-block loop exits from another point in the loop;
     * therefore, the tail is not always executed.
     */
    return false;
  cnt = 0;
  for (p = FG_PRED(LP_HEAD(lp)); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (FG_LOOP(PSI_NODE(p)) == lp)
      cnt++;
    if (cnt > 1)
      return false;
  }
  return true;
}

/** \brief Build the natural loop and loop region given the head and tail flow
 * graph nodes.
 *
 * The natural loop is represented by a linked list headed by the variable
 * naturalloop.  As nodes are visited, they are added to this list.
 * The natural loop is built by first adding the loop head to the list
 * and then visiting the tail of the loop and recursively visiting its
 * predecessors.  The natural loop contains the nodes of any nested loops.
 * Note that if there are other retreating edges whose head values are
 * identical to the head of the loop, these extra edges are taken into
 * consideration so that the loops defined by retreating edges with identical
 * head values are combined into a single loop.
 *
 * The loop region is defined to be the set of nodes in the natural loop
 * which do not belong to any nested loops.  The loop region is represented
 * as a linked list headed by the field LP_FG of the current loop.  The nodes
 * which belong to the current loop region are linked together using the
 * FG_NEXT fields.  The field FG_LOOP records the loop to which it
 * belongs.  Also, the loop's LP_CALLFG and LP_NOBLA fields are set if
 * the respective criteria are met.
 *
 * NOTE that build_loop is used for two purposes:
 * 1.  to aid in determining the level and parent values of the loops
 *     so that the loops can be sorted,
 * 2.  to build the regions of the loops after the loops have been sorted.
 * For the first case, the natural loop list and loop region list are
 * identical.
 */
static void
build_loop(int loop)
{
  int head;
  int edge;

  /* begin building the loop by adding the loop head to the region */

  LP_FG(loop) = head = LP_HEAD(loop);
  LP_CALLFG(loop) = BIH_EX(FG_TO_BIH(head));
#ifdef BIH_NOBLA
  LP_NOBLA(loop) = BIH_NOBLA(FG_TO_BIH(head));
#endif
  LP_JMP_TBL(loop) = FG_JMP_TBL(head);
#ifdef FE90
  LP_PARLOOP(loop) = FG_PAR(head);
#else
  LP_PARLOOP(loop) = BIH_PARLOOP(FG_TO_BIH(head));
#endif
#ifdef LP_PARALN
  LP_PARALN(loop) = BIH_PARALN(FG_TO_BIH(head));
#endif

  /* init linked list of regions assigned to this loop */
  FG_LOOP(head) = loop;
  FG_NEXT(head) = 0;

  /* init linked list of regions in this naturalloop */
  naturalloop = head;
  FG_NATNXT(head) = 0;
  FG_VISITED(head) = 1;

  current_lp = loop;
  current_lp_tail = head;
  /*
   * repeat for the tail of the loop and all of its predecessors;
   * also, include the predecessors of any back edges whose
   * head values are identical.
   */
  add_to_region((int)LP_TAIL(loop));
  edge = EDGE_NEXT(LP_EDGE(loop));
  while (edge >= 0) {
    add_to_region((int)EDGE_PRED(edge));
    edge = EDGE_NEXT(edge);
  }
}

/** \brief Add a node to the loop region
 */
static void
add_to_region(int v)
{
  PSI_P p;

  if (FG_VISITED(v) == 1)
    return;

  /* add v to nodes in naturalloop */
  FG_VISITED(v) = 1;
  FG_NATNXT(v) = naturalloop;
  naturalloop = v;

  if (lp_topsort) {
    /* first recurse on non-fall-through edges.
     * then do the fall-through edges.
     * this tends to keep the fall-through edges adjacent. */
    for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      if (!PSI_FT(p))
        add_to_region((int)PSI_NODE(p));
    }
    for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      if (PSI_FT(p))
        add_to_region((int)PSI_NODE(p));
    }
  }

  if (FG_LOOP(v) == 0) {
    /* if no enclosed loop has grabbed this node, */
    /* add v to current_lp's assigned regions. */

    /* add to linked list of assigned regions */
    if (lp_topsort) {
      FG_NEXT(v) = 0;
      FG_NEXT(current_lp_tail) = v;
      current_lp_tail = v;
    } else {
      FG_NEXT(v) = LP_FG(current_lp);
      LP_FG(current_lp) = v;
    }

    FG_LOOP(v) = current_lp;
    if (BIH_EX(FG_TO_BIH(v)))
      LP_CALLFG(current_lp) = 1;
#ifdef BIH_NOBLA
    if (BIH_NOBLA(FG_TO_BIH(v)))
      LP_NOBLA(current_lp) = 1;
#endif
    if (FG_JMP_TBL(v))
      LP_JMP_TBL(current_lp) = 1;
  }
  /* recurse for the predecessors of v  */
  if (!lp_topsort) {
    for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
      add_to_region((int)PSI_NODE(p));
    }
  }

}

/*********************************************************************/

static void
add_lpexit(int lpx, int exit)
{
  PSI_P p;
  bool mult;

  mult = false;
  for (p = LP_EXITS(lpx); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    mult = true;
    if (PSI_NODE(p) == exit)
      goto set_mexits;
  }
  if (OPTDBG(9, 8))
    fprintf(gbl.dbgfil, "   loop %d exits at %d\n", lpx, exit);
  GET_PSI(p);
  PSI_NODE(p) = exit;
  PSI_NEXT(p) = LP_EXITS(lpx);
  PSI_ALL(p) = 0;
  LP_EXITS(lpx) = p;

set_mexits:
  LP_MEXITS(lpx) = mult;

}

/** \brief Unvisit blocks in the natural loop created by build_loop; also
 * optionally unvisit blocks in the loop region.
 */
static void
unvisit(int list, int clearfgloop)
{
  int i;

  for (i = list; i != 0; i = FG_NATNXT(i)) {
    FG_VISITED(i) = 0;
    if (clearfgloop)
      FG_LOOP(i) = 0;
  }
}

static int malformed_loop;

/** Build the natural loop and loop region given the head and tail flow
 * graph nodes and mark the nodes in the loop as 'malformed' (FG_MALF_LP).
 * Make sure the nodes of any enclosed loop are no set.
 */
static void
malformed(int head, int tail)
{
  if (OPTDBG(9, 8)) {
    fprintf(gbl.dbgfil, "MALFORMED loop %s:%s:%d\n", gbl.src_file,
            getprint((int)BIH_LABEL(gbl.entbih)), BIH_LINENO(FG_TO_BIH(head)));
  }

  FG_VISITED(head) = 1;
  FG_NATNXT(head) = 0;
  malformed_loop = head;
  /*
   * repeat for the tail of the loop and all of its predecessors;
   * also, include the predecessors of any back edges whose
   * head values are identical.
   */
  add_to_malf(tail);
  {
    int i;
    int nxt;
    /*
     * the natural loop is malformed -- unvisit the nodes in the loop and
     * mark each fg
     */
    i = malformed_loop;
    while (i) {
      nxt = FG_NATNXT(i);
      FG_VISITED(i) = 0;
      FG_NATNXT(i) = 0;
      if (FG_LOOP(head) == FG_LOOP(i))
        FG_MALF_LP(i) = 1;
      i = nxt;
    }
  }
}

static void
add_to_malf(int v)
{
  PSI_P p;

  if (FG_VISITED(v) == 1)
    return;

  /* add v to nodes in naturalloop */
  FG_VISITED(v) = 1;
  FG_NATNXT(v) = malformed_loop;
  malformed_loop = v;

  for (p = FG_PRED(v); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    add_to_malf((int)PSI_NODE(p));
  }
}

#ifndef FE90
/** \brief A routine to find loops whose control flow follows the pattern of a
 * "while" loop.
 *
 * A "while _cond_" loop is changed to "if _cond_ do ... while !_cond_".  A
 * "for" loop is a form of a "while" loop.
 */
static void
convert_loop(int loop)
{
  int head, tail;
  int headbih, tailbih, iltx, newhead, exit;
  SPTR label, tmpsptr;
  int tmp, bihx, lastheadbih, fgx;
  int oldheadbih;
  int i, br_ilt;
  PSI_P p, q;
  int new_tree;

  if (XBIT(6, 0x80000000))
    return;
  head = LP_HEAD(loop);
  headbih = FG_TO_BIH(head);
  /*
   * if expander has marked a Doloop as zerotrip or if a previous call
   * to findloop on this function (i.e., by the vectorizer) has converted
   * a loop, pass along this info and return.
   */
  if (BIH_ZTRP(headbih) == 1) {
    LP_ZEROTRIP(loop) = 1;
    return;
  }

  /* check if head contains <= ILT_THRESH ILTs;
   * do nothing if the loop only has one block  */
  if (head == (tail = LP_TAIL(loop)))
    return;

  /* label of head does not exist or it is used more than once  */

  if ((label = BIH_LABEL(headbih)) == 0 || RFCNTG(label) != 1) {
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil, "---convert_loop(%d): bih%d wrong label %d\n", loop,
              headbih, label);
    return;
  }
/* check if head contains <= ILT_THRESH ILTs  */
/* note that the 'head' may be more than one block in some cases */

#define ILT_THRESH 4

  i = 0;
  for (bihx = headbih; 1; bihx = BIH_NEXT(bihx)) {
    lastheadbih = bihx;
    fgx = BIH_TO_FG(bihx);
    if (FG_LOOP(fgx) != loop)
      return;
    iltx = BIH_ILTFIRST(bihx);
    if (ILT_DBGLINE(iltx))
      iltx = ILT_NEXT(iltx);
    for (tmp = iltx; tmp; tmp = ILT_NEXT(tmp))
      ++i;

    if (i > ILT_THRESH) {
      if (OPTDBG(9, 8))
        fprintf(gbl.dbgfil, "---convert_loop(%d): bih%d > %d ILT\n", loop, bihx,
                ILT_THRESH);
      return;
    }
    br_ilt = BIH_ILTLAST(bihx);
    /* if there are two successors of bihx, this is the branch.
     * if there is one successor of bihx and it is a fall-through,
     * go around this loop again. */
    p = FG_SUCC(fgx);
    if (PSI_NEXT(p) == PSI_P_NULL) {
      /* only one successor */
      int target;
      target = PSI_NODE(p);
      if (FG_LOOP(target) != loop || FG_TO_BIH(target) != BIH_NEXT(bihx) ||
          (br_ilt && IL_TYPE(ILI_OPC(ILT_ILIP(br_ilt))) == ILTY_BRANCH)) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil, "---convert_loop(%d): bih%d single target with "
                              "jump or not fall through\n",
                  loop, bihx);
        return;
      }
    } else {
      /* Branch at br_ilt needs to be a conditional to avoid crashing
       * below when its condition code is reversed.
       */
      if (br_ilt <= 0 ||
          !(is_integer_comparison_opcode(ILI_OPC(ILT_ILIP(br_ilt))) ||
            is_floating_comparison_opcode(ILI_OPC(ILT_ILIP(br_ilt))))) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil,
                  "---convert_loop(%d): bih%d missing conditional jump\n", loop,
                  bihx);
        return;
      }
      /*
       * there can only be two successors of head -- the first one is the one
       * which the head falls through to and the second is the one which the
       * head exits to.
       */
      newhead = PSI_NODE(p);
      if (FG_LOOP(newhead) != loop) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil,
                  "---convert_loop(%d): first succ, %d, not in loop\n", loop,
                  newhead);
        return;
      }
      p = PSI_NEXT(p);
      exit = PSI_NODE(p);
      if (FG_LOOP(exit) == loop) {
        if (OPTDBG(9, 8))
          fprintf(
              gbl.dbgfil,
              "---convert_loop(%d) succ node, %d, of head, %d, is in loop\n",
              loop, exit, fgx);
        return;
      }
      if (PSI_NEXT(p) != PSI_P_NULL) {
        if (OPTDBG(9, 8))
          fprintf(gbl.dbgfil, "---convert loop(%d): >2 succ(h) of head ,%d\n",
                  loop, fgx);
        return;
      }
      break; /* found the loop exit branch */
    }
  }

  /*
   * the next physical block of the tail must be the block which the head
   * exits to
   */
  tailbih = FG_TO_BIH(tail);
  if (BIH_NEXT(tailbih) != FG_TO_BIH(exit)) {
    return;
  }
  /* the number of successors of tail is 1  */

  if (PSI_NEXT(FG_SUCC(tail)) != PSI_P_NULL) {
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil,
              "---convert_loop(%d) more than one success of tail %d\n", loop,
              tail);
    return;
  }
  assert(ILI_OPC(ILT_ILIP(BIH_ILTLAST(tailbih))) == IL_JMP,
         "convert_loop: tail not IL_JMP", loop, ERR_Severe);

  if (BIH_PAR(FG_TO_BIH(newhead)) != BIH_PAR(tailbih)) {
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil,
              "---convert_loop(%d): BIH_PAR of %d and %d do not match\n", loop,
              FG_TO_BIH(newhead), tailbih);
    return;
  }
  if (BIH_TASK(FG_TO_BIH(newhead)) != BIH_TASK(tailbih)) {
    if (OPTDBG(9, 8))
      fprintf(gbl.dbgfil,
              "---convert_loop(%d): BIH_TASK of %d and %d do not match\n", loop,
              FG_TO_BIH(newhead), tailbih);
    return;
  }

  /*****  loop can be converted  *****/

  if (OPTDBG(9, 8))
    fprintf(gbl.dbgfil, "---convert_loop(%d), newhead:%d, tail:%d, exit:%d\n",
            loop, newhead, tail, exit);

  BIH_LABEL(headbih) = SPTR_NULL; /* old head no longer has a label  */

  /*
   * check if the new head (the block following old head) already has a
   * label.  If so, use it.
   */
  oldheadbih = headbih;
  headbih = FG_TO_BIH(newhead);
  tmpsptr = BIH_LABEL(headbih);
  if (tmpsptr != SPTR_NULL) {
    RFCNTD(label);
    RFCNTI(tmpsptr);
    label = tmpsptr;
  } else {
    BIH_LABEL(headbih) = label;
    ILIBLKP(label, headbih);
  }

  /*
   * get the tail block and add the ilts from the old head block (except
   * for the last ilt) to the tail after the ilt which precedes its branch
   */
  rdilts(tailbih); /* sets bihb.callfg           */

  tmp = ILT_PREV(BIH_ILTLAST(tailbih)); /* where to begin adding ilts */

  /*
   * go through and add the ilts from head blocks which occur before the branch
   * to
   * just before the last ilt in the tail.  iltx is the first ilt which has
   * to be added.
   * NOTE: can't simply reuse the ili for the ilt;  problem
   * occurs if a jsr appears in the tree -- by definition,
   * jsr's aren't shared unless an explicit cse situation
   * occurs.  To ensure jsr's are rewritten, use the rewrite
   * ili mechanism using the IL_NULL as the 'old' & 'new'
   * arguments.
   */
  for (bihx = oldheadbih; 1; bihx = BIH_NEXT(bihx)) {
    iltx = BIH_ILTFIRST(bihx);
    if (ILT_DBGLINE(iltx))
      iltx = ILT_NEXT(iltx);
    for (; iltx && iltx != br_ilt; iltx = ILT_NEXT(iltx)) {
      int ilix = ILT_ILIP(iltx);

      /* we don't want IL_LABEL that marks beggining of a scope to be moved to
       * last block of loop
       */
      if (ilix && ILI_OPC(ilix) == IL_LABEL && BEGINSCOPEG(ILI_OPND(ilix, 1)))
        continue;
      new_tree = rewr_ili(ilix, 1, 1);
      tmp = addilt(tmp, new_tree); /* updates bihb.callfg  */
      if (OPTDBG(9, 8))
        fprintf(gbl.dbgfil, "---convert_loop(%d), oldtree:%d, newtree:%d\n",
                loop, ILT_ILIP(iltx), new_tree);
    }
    BIH_QJSR(tailbih) |= BIH_QJSR(bihx);
    bihb.callfg |= BIH_EX(bihx);
    if (bihx == lastheadbih)
      break;
  }

  /*
   * Since the last ilt is one which branches back to the old head, change
   * that branch to the complement of the one which is in the old head. The
   * label referenced is the one which labels the new head
   */
  tmp = ILT_NEXT(tmp);
  assert(tmp == BIH_ILTLAST(tailbih), "convert_loop: wrong last ilt", tmp, ERR_Severe);
  new_tree = rewr_ili((int)ILT_ILIP(br_ilt), 1, 1);
  ILT_ILIP(tmp) = compl_br((int)new_tree, label);
  if (OPTDBG(9, 8))
    fprintf(gbl.dbgfil, "---convert_loop(%d), oldbrtree:%d, newbrtree:%d\n",
            loop, ILT_ILIP(br_ilt), ILT_ILIP(tmp));
  ILT_EX(tmp) = ILT_EX(br_ilt);
  BIH_EX(tailbih) = bihb.callfg | ILT_EX(tmp);
  wrilts(tailbih);
  rewr_cln_ili();

  /*
   * remove old head nodes from the current loop and define the new head of the
   * loop
   */

  for (bihx = oldheadbih; 1; bihx = BIH_NEXT(bihx)) {
    fgx = BIH_TO_FG(bihx);
    rm_node_from_loop(fgx);
    FG_LOOP(fgx) = 0;
    FG_NEXT(fgx) = 0;
    if (bihx == lastheadbih)
      break;
  }

  LP_HEAD(loop) = newhead; /* already in the region  */
  FG_NEXT(head) = 0;       /* note: head is the last block in the  */
  FG_NEXT(tail) = 0;       /* region and tail is the next to last */

  /* replace head in the successor list of tail with exit */

  for (p = FG_SUCC(tail); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (PSI_NODE(p) == head) {
      PSI_NODE(p) = exit;
      PSI_FT(p) = 1; /* tail now falls thru */
      break;
    }
  }
  assert(p != PSI_P_NULL, "convert_loop: head not succ of tail", tail, ERR_Severe);

  /*
   * remove tail from the predecessor list of head and add tail to the
   * predecessor list of exit
   */
  q = PSI_P_NULL;
  for (p = FG_PRED(head); p != PSI_P_NULL; p = PSI_NEXT(p)) {
    if (PSI_NODE(p) == tail) { /* item is for the node tail */
      if (q == PSI_P_NULL)
        FG_PRED(head) = PSI_NEXT(p);
      else
        PSI_NEXT(q) = PSI_NEXT(p);
      PSI_NEXT(p) = FG_PRED(exit); /* add the item to pred(exit) */
      FG_PRED(exit) = p;
      break;
    }
    q = p;
  }
  assert(p != PSI_P_NULL, "convert_loop: tail not pred of head", head, ERR_Severe);
  BIH_FT(tailbih) = 1;

  /*
   * add newhead to the successor list of tail and add tail to the
   * precessor list of newhead
   */
  (void)add_succ(tail, newhead);
  (void)add_pred(newhead, tail);

  /*
   * mark this loop as one which has been converted -- the flag is zerotrip
   * meaning that there exists a test prior to the loop which may prevent
   * the loop from being executed. This flag may be used by induction.
   */
  LP_ZEROTRIP(loop) = 1;
  BIH_ZTRP(headbih) = 1;
  /*
   * the line number of the new loop head should be the line number of the
   * the old loop head instead of the new head's current line number.
   * necessary to maintain consistency with loop-scoped pragmas/directives.
   */
  BIH_LINENO(headbih) = BIH_LINENO(FG_TO_BIH(head));

#if DEBUG
  if (DBGBIT(10, 2)) {
    fprintf(gbl.dbgfil, "after convert loop, bihs: head %d, tail %d\n", headbih,
            tailbih);
    dmpilt(headbih);
    if (headbih != tailbih)
      dmpilt(tailbih);
  }
#endif

}
#else
/** \brief Routine to find loops whose control flow follows the pattern of a
 * "while" loop.
 *
 * A "while <cond>" loop is changed to "if <cond> do ... while !<cond>".  A
 * "for" loop is a form of a "while" loop.
 */
static void
convert_loop(int loop)
{
}
#endif

/*******************************************************************/
/*
 * reorder the loops in the LP_LOOP order.
 * right now they are top-sorted according to the parent relationship,
 * but otherwise unordered.  Here we use a more constructive sort,
 * so a loop's children are contiguous to the loop.
 */

static void
addloop(int l, int *pn)
{
  int ll;
  for (ll = LP_CHILD(l); ll; ll = LP_SIBLING(ll)) {
    addloop(ll, pn);
  }
  ++(*pn);
  LP_LOOP(*pn) = l;
} /* addloop */

void
reorderloops()
{
  int n, l;
  n = 0;
  for (l = LP_CHILD(0); l; l = LP_SIBLING(l)) {
    addloop(l, &n);
  }
#if DEBUG
  if (n != opt.nloops) {
    interr("reorderloops: wrong number of loops", n, ERR_Severe);
  }
#endif
} /* reorderloops */

/*
 * reinsert each loop into its parent loop child list
 * so the order of the loop children matches the order in
 * the BIH list.  Similarly, reinsert each FG node into its
 * loop FG list so the FG list matches the order in the BIH list.
 */
void
sortloops()
{
  int lpx, fgx, bihx, px;
  /* clear the LP_CHILD links */
  for (lpx = 0; lpx <= opt.nloops; ++lpx) {
    LP_CHILD(lpx) = 0;
    LP_FG(lpx) = 0;
    LP_SIBLING(lpx) = 0;
  }
  /* clear FG_NEXT link */
  for (fgx = 1; fgx < opt.num_nodes; ++fgx) {
    FG_NEXT(fgx) = 0;
  }
  for (bihx = gbl.entbih; bihx; bihx = BIH_NEXT(bihx)) {
    fgx = BIH_TO_FG(bihx);
    lpx = FG_LOOP(fgx);
    FG_NEXT(fgx) = LP_FG(lpx);
    LP_FG(lpx) = fgx; /* insert in reverse order, reverse later */
    if (lpx && fgx == LP_HEAD(lpx)) {
      px = LP_PARENT(lpx);
      LP_SIBLING(lpx) = LP_CHILD(px);
      LP_CHILD(px) = lpx;
    }
  }
  for (lpx = 0; lpx <= opt.nloops; ++lpx) {
    /* reverse LP_CHILD links, FG_NEXT */
    int childx, nextchildx, newchildx;
    int nextfgx, newfgx;
    childx = LP_CHILD(lpx);
    newchildx = 0;
    for (; childx; childx = nextchildx) {
      nextchildx = LP_SIBLING(childx);
      LP_SIBLING(childx) = newchildx;
      newchildx = childx;
    }
    LP_CHILD(lpx) = newchildx;
    fgx = LP_FG(lpx);
    newfgx = 0;
    for (; fgx; fgx = nextfgx) {
      nextfgx = FG_NEXT(fgx);
      FG_NEXT(fgx) = newfgx;
      newfgx = fgx;
    }
    LP_FG(lpx) = newfgx;
  }
} /* sortloops */

/* Query whether lp1 is a child loop of lp2 */

bool
is_childloop(int lp1, int lp2)
{
  int lp_sib;

  for (lp_sib = LP_CHILD(lp2); lp_sib; lp_sib = LP_SIBLING(lp_sib)) {
    if ((lp1 == lp_sib) || is_childloop(lp1, lp_sib))
      return true;
  }

  return false;
}

/*
 * return true if loop lp1 contains loop lp2
 * by convention, loop zero contains all loops, and a loop contains itself
 */
bool
contains_loop(int lp1, int lp2)
{
  if (lp1 == 0 || lp1 == lp2)
    return true;
  if (lp2 == 0)
    return false;
  while (LP_LEVEL(lp2) > LP_LEVEL(lp1))
    lp2 = LP_PARENT(lp2);
  if (lp2 == lp1)
    return true;
  return false;
} /* contains_loop */

/*
 * Return true if the loops lp1 and lp2 are overlapping.
 * By convention, loop zero overlaps all loops.
 */
bool
overlapping_loops(int lp1, int lp2)
{
  if (lp1 == 0 || lp2 == 0)
    return true;
  if (LP_LEVEL(lp1) < LP_LEVEL(lp2))
    return contains_loop(lp1, lp2);
  return contains_loop(lp2, lp1);
}

/*******************************************************************/

void
__dump_region(FILE *ff, int lp)
{
  int j, v;

  j = 0;
  fprintf(ff, "       region(%d):", lp);
  for (v = LP_FG(lp); v != 0; v = FG_NEXT(v)) {
    if (j == 9) {
      fprintf(ff, "\n               ");
      j = 0;
    }
    fprintf(ff, " %-5d", v);
    j++;
  }
  fprintf(ff, "\n");
}

/** \brief Dump a loop region
 */
void
dump_region(int lp)
{
  __dump_region(gbl.dbgfil, lp);
}

/*******************************************************************/

void
__dump_loop(FILE *ff, int lp)
{
  static char star[] = {' ', '*'};
  static char zt[] = {' ', '0'};
  int i;
  PSI_P p;
  Q_ITEM *q;

  i = lp;
  if (ff == NULL)
    ff = stderr;

  fprintf(ff, "%5d.%c%c level: %-5d  parent: %-5d  fg: %-5d", i,
          star[LP_INNERMOST(i)], zt[LP_ZEROTRIP(i)], LP_LEVEL(i), LP_PARENT(i),
          LP_FG(i));
  if (i)
    fprintf(ff, "  edge: (%d, %d)", LP_TAIL(i), LP_HEAD(i));
  fprintf(ff, "\n");

  fprintf(ff, "       child: %-5d  sibling: %-5d  count: %-5d", LP_CHILD(i),
          LP_SIBLING(i), LP_COUNT(i));
  if (i) {
    fprintf(ff, "  lineno: (%d, %d)", BIH_LINENO(FG_TO_BIH(LP_TAIL(i))),
            BIH_LINENO(FG_TO_BIH(LP_HEAD(i))));
  }
  fprintf(ff, "\n");

  fprintf(ff, "       %s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
          LP_INNERMOST(i) ? "<inner>" : "", LP_CALLFG(i) ? "<call>" : "",
          LP_NOBLA(i) ? "<nobla>" : "", LP_QJSR(i) ? "<qjsr>" : "",
          LP_MEXITS(i) ? "<mexits>" : "", LP_JMP_TBL(i) ? "<jmp_tbl>" : "",
          LP_PARLOOP(i) ? "<parloop>" : "", LP_CS(i) ? "<cs>" : "",
          LP_CSECT(i) ? "<csect>" : "", LP_PARREGN(i) ? "<parregn>" : "",
          LP_PARSECT(i) ? "<parsect>" : "", LP_XTNDRNG(i) ? "<xtndrng>" : "",
#ifdef LP_SMOVE
          LP_SMOVE(i) ? "<smove>" : "",
#else
          "",
#endif
#ifdef LP_PARALN
          LP_PARALN(i) ? "<paraln>" : "",
#else
          "",
#endif
#ifdef LP_INLNPTR
          LP_INLNPTR(i) ? "<inlnptr>" : "",
#else
          "",
#endif
          LP_CNCALL(i) ? "<cncall>" : "", LP_TASK(i) ? "<task>" : "",
          LP_TAIL_AEXE(i) ? "<tailaexe>" : "", LP_VOLLAB(i) ? "<vollab>" : "");
  if (i) {
#ifdef FE90
    fprintf(ff, "       %s%s", LP_FORALL(i) ? "<forall>" : "",
            LP_MASTER(i) ? "<master>" : "");
#endif
    fprintf(ff, "\n");
    fprintf(ff, "       exits:");
    for (p = LP_EXITS(i); p != PSI_P_NULL; p = PSI_NEXT(p))
      fprintf(ff, " %d", PSI_NODE(p));
  }
  q = LP_STL_PAR(i);
  if (q) {
    fprintf(ff, "\n       stl_par:");
    for (; q != NULL; q = q->next) {
      fprintf(ff, " %d(", (int)q->info);
      (void)__print_nme(ff, (int)q->info);
      fprintf(ff, ")");
    }
  }
  fprintf(ff, "\n");
  __dump_region(ff, i);

}

/** \brief Dump the loop table
 */
void
dump_loops(void)
{
  static char star[] = {' ', '*'};
  int i, j, v;
  fprintf(gbl.dbgfil, "\n\n*****  Loops (%d) for Function \"%s\"  *****\n",
          opt.nloops, getprint(BIH_LABEL(gbl.entbih)));

  for (i = 0; i <= opt.nloops; i++) {
    __dump_loop(gbl.dbgfil, i);
  }

  if (opt.nloops) {
    fprintf(gbl.dbgfil, "\n   sorted order:");
    j = 0;
    for (i = 1; i <= opt.nloops; i++) {
      if (j == 9) {
        j = 0;
        fprintf(gbl.dbgfil, "\n                ");
      }
      v = LP_LOOP(i);
      fprintf(gbl.dbgfil, "  %c%-5d", star[LP_INNERMOST(v)], v);
      j++;
    }
    if (j)
      fprintf(gbl.dbgfil, "\n");
  }

}
