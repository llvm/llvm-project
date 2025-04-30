/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
   \file
   \brief Do "points to" analysis; save information in NME and PTE structures
 */
/*
 * ###
 * handle struct member LHS
 * check that function arguments and return value are handled properly
 *
 * F90 - pointer arguments handled properly, also pointer return values via IPA
 * F90 - IPA initial values
 */

#include "gbldefs.h"
#include "global.h"
#include "error.h"
#include "symtab.h"
#include "nme.h"
#include "ast.h"
#include "gramtk.h"
#include "pd.h"
#include "optimize.h"

#define TRACEFLAG 10
#define TRACEBIT 0x100000
#define TRACESTRING "pte-"
#include "trace.h"

/*
 * The data structures used:
 * . temporary pointer descriptor (TPD) structure, used while processing a
 *   single statement.  This is used to build the subsequent data structures.
 *   descriptor type, symbol table pointer, integer value
 * . pointer source descriptor (PSD), used to describe the pointers assigned
 *   in the program.  This is quite like a persistent PD; the number of these
 *   is the 'width' of the data flow value carried between blocks.
 * . pointer assignment (PA), describes the LHS with a PSD, and the RHS
 *   with either another PSD, for assignment like p=q, or ++p, or with
 *   a pointer target entry if the RHS is a constant address, like p=&a
 *   or p=NULL.
 *
 *  tpdx = index to temporary pointer descriptor
 *  psdx = index to pointer source descriptor
 *  ptex = index to pointer target entry
 *  atex = index to actual target entry
 *  asx  = index to assignment table
 *  sl   = slot number
 *  sptr = symbol table pointer
 *  bihx = block table index
 *  ilix = ili table index
 *  iltx = ilt table index
 *  nmex = nme table index
 */

/*
 * This is a temporary pointer descriptor (TPD) data structure, to hold
 * addresses
 * while processing a single statement
 */
typedef struct {
  int type;
  int sym;
  int val;
  int link;
} TPD;

static STG_DECLARE(gtpd, TPD) = { STG_INIT };

#if DEBUG
#define TTYPE(tpdx)                                            \
  gtpd.stg_base[tpdx >= 0 && tpdx < gtpd.stg_avail             \
                    ? tpdx                                     \
                    : (interr("bad TTYPE index", tpdx, 4), 0)] \
      .type
#define TSYM(tpdx)                                            \
  gtpd.stg_base[tpdx >= 0 && tpdx < gtpd.stg_avail            \
                    ? tpdx                                    \
                    : (interr("bad TSYM index", tpdx, 4), 0)] \
      .sym
#define TVAL(tpdx)                                            \
  gtpd.stg_base[tpdx >= 0 && tpdx < gtpd.stg_avail            \
                    ? tpdx                                    \
                    : (interr("bad TVAL index", tpdx, 4), 0)] \
      .val
#define TLINK(tpdx)                                            \
  gtpd.stg_base[tpdx >= 0 && tpdx < gtpd.stg_avail             \
                    ? tpdx                                     \
                    : (interr("bad TLINK index", tpdx, 4), 0)] \
      .link
#else
#define TTYPE(tpdx) gtpd.stg_base[tpdx].type
#define TSYM(tpdx) gtpd.stg_base[tpdx].sym
#define TVAL(tpdx) gtpd.stg_base[tpdx].val
#define TLINK(tpdx) gtpd.stg_base[tpdx].link
#endif

/* unknown target */
#define TT_UNINIT 0x100
/* unknown target */
#define TT_UNK 0x110
/* unknown global ... used for type target or for dummy pointer source */
#define TT_GLOB 0x120
/* precise symbol */
#define TT_PSYM 0x130
/* imprecise symbol */
#define TT_ISYM 0x140
/* anonymous memory, from IPA */
#define TT_ANON 0x150
/* dynamically allocated memory, also from IPA */
#define TT_GDYN 0x160
/* imprecise dynamically allocated memory, also from IPA */
#define TT_IGDYN 0x170
/* dynamically allocated memory, from this function call */
#define TT_LDYN 0x180
/* imprecise dynamically allocated memory, from this function call */
#define TT_ILDYN 0x190
/* constant address, used for NULL */
#define TT_CON 0x1a0
/* indirection */
#define TT_IND 0x1b0
/* imprecise indirection */
#define TT_IIND 0x1c0
/* member */
#define TT_MEM 0x1d0
/* nonlocal memory */
#define TT_NLOC 0x1e0

#define TTE_NULL -1

/* MUSTFLAG is set if this is a must point to situation
   IMPFLAG is set if this is an imprecise descriptor */
#define TTMUSTFLAG 1
#define TTIMPFLAG 2
/* remove MUST flag */
#define TTMASKMUST(typ) ((typ)&0xfffe)
/* remove MUST and IMP flags */
#define TTMASK(typ) ((typ)&0xfffd)
#define TTMUST(typ) ((typ)&TTMUSTFLAG)
#define TTIMP(typ) ((typ)&TTIMPFLAG)

/*
 * this is a temporary pointer source descriptor (PSD)
 * it looks a lot like NMEs (but no subscripts, yes struct members)
 */
typedef struct {
  int type;
  int parent;
  int sym;
  int next;  /* next PSD with same parent */
  int child; /* child PSD */
  int slot;
  int hashlk;
} PSD;

static struct {
  STG_MEMBERS(PSD);
  int *slot;
  int nslots;
} gpsd = {STG_INIT, NULL, 0};

#if DEBUG
#define PSD_TYPE(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail                \
                    ? psdx                                        \
                    : (interr("bad PSD_TYPE index", psdx, 4), 0)] \
      .type
#define PSD_PARENT(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail                  \
                    ? psdx                                          \
                    : (interr("bad PSD_PARENT index", psdx, 4), 0)] \
      .parent
#define PSD_SYM(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail               \
                    ? psdx                                       \
                    : (interr("bad PSD_SYM index", psdx, 4), 0)] \
      .sym
#define PSD_CHILD(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail                 \
                    ? psdx                                         \
                    : (interr("bad PSD_CHILD index", psdx, 4), 0)] \
      .child
#define PSD_NEXT(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail                \
                    ? psdx                                        \
                    : (interr("bad PSD_NEXT index", psdx, 4), 0)] \
      .next
#define PSD_HASHLK(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail                  \
                    ? psdx                                          \
                    : (interr("bad PSD_HASHLK index", psdx, 4), 0)] \
      .hashlk
#define PSD_SLOT(psdx)                                            \
  gpsd.stg_base[psdx >= 0 && psdx < gpsd.stg_avail                \
                    ? psdx                                        \
                    : (interr("bad PSD_SLOT index", psdx, 4), 0)] \
      .slot
#define SLOT(sl)                         \
  gpsd.slot[sl >= 0 && sl <= gpsd.nslots \
                ? sl                     \
                : (interr("bad SLOT index", sl, 4), 0)]
#else
#define PSD_TYPE(psdx) gpsd.stg_base[psdx].type
#define PSD_PARENT(psdx) gpsd.stg_base[psdx].parent
#define PSD_SYM(psdx) gpsd.stg_base[psdx].sym
#define PSD_CHILD(psdx) gpsd.stg_base[psdx].child
#define PSD_NEXT(psdx) gpsd.stg_base[psdx].next
#define PSD_HASHLK(psdx) gpsd.stg_base[psdx].hashlk
#define PSD_SLOT(psdx) gpsd.stg_base[psdx].slot
#define SLOT(sl) gpsd.slot[sl]
#endif

/* pointer source descriptor hash table */
#define PSDHSZ 127

static int psdhsh[PSDHSZ];

/*
 * This is a table of actual pointer target entries (APTE), hashed lookup
 */
typedef struct {
  int hashlk;
  int type;
  int sym;
  int val;
  unsigned int mark;
  int stride;
} APTE;
#define APTEHSZ 521
static STG_DECLARE(apte, APTE) = { STG_INIT };
static int aptehsh[APTEHSZ];

#if DEBUG
#define APTE_HASHLK(aptex)                                            \
  apte.stg_base[aptex >= 0 && aptex < apte.stg_avail                  \
                    ? aptex                                           \
                    : (interr("bad APTE_HASHLK index", aptex, 4), 0)] \
      .hashlk
#define APTE_TYPE(aptex)                                            \
  apte.stg_base[aptex >= 0 && aptex < apte.stg_avail                \
                    ? aptex                                         \
                    : (interr("bad APTE_TYPE index", aptex, 4), 0)] \
      .type
#define APTE_SPTR(aptex)                                            \
  apte.stg_base[aptex >= 0 && aptex < apte.stg_avail                \
                    ? aptex                                         \
                    : (interr("bad APTE_SPTR index", aptex, 4), 0)] \
      .sym
#define APTE_VAL(aptex)                                            \
  apte.stg_base[aptex >= 0 && aptex < apte.stg_avail               \
                    ? aptex                                        \
                    : (interr("bad APTE_VAL index", aptex, 4), 0)] \
      .val
#define APTE_STRIDE(aptex)                                            \
  apte.stg_base[aptex >= 0 && aptex < apte.stg_avail                  \
                    ? aptex                                           \
                    : (interr("bad APTE_STRIDE index", aptex, 4), 0)] \
      .stride
#define APTE_MARK(aptex)                                            \
  apte.stg_base[aptex >= 0 && aptex < apte.stg_avail                \
                    ? aptex                                         \
                    : (interr("bad APTE_MARK index", aptex, 4), 0)] \
      .mark
#else
#define APTE_HASHLK(aptex) apte.stg_base[aptex].hashlk
#define APTE_TYPE(aptex) apte.stg_base[aptex].type
#define APTE_SPTR(aptex) apte.stg_base[aptex].sym
#define APTE_VAL(aptex) apte.stg_base[aptex].val
#define APTE_STRIDE(aptex) apte.stg_base[aptex].stride
#define APTE_MARK(aptex) apte.stg_base[aptex].mark
#endif

/*
 * This is a temporary pointer target entry (TPTE) table, with a free list
 */
typedef struct {
  int next;  /* next pointer target */
  int aptex; /* actual pointer target */
#if DEBUG
  int flag; /* used for debugging */
#endif
} TPTE;
static struct {
  STG_MEMBERS(TPTE);
  int xstg_free, save_stg_avail, save_stg_free;
} gpte = {STG_INIT, -1, 0, 0};

#define TPTE_UNK 0
#define TPTE_UNINIT 1
#define TPTE_NULL -1

#if DEBUG
#define TPTE_NEXT(ptex)                                            \
  gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                 \
                    ? ptex                                         \
                    : (interr("bad TPTE_NEXT index", ptex, 4), 0)] \
      .next
#define TPTE_APTEX(ptex)                                            \
  gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                  \
                    ? ptex                                          \
                    : (interr("bad TPTE_APTEX index", ptex, 4), 0)] \
      .aptex
#define TPTE_TYPE(ptex)                                                      \
  APTE_TYPE(gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                 \
                              ? ptex                                         \
                              : (interr("bad TPTE_TYPE index", ptex, 4), 0)] \
                .aptex)
#define TPTE_SPTR(ptex)                                                      \
  APTE_SPTR(gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                 \
                              ? ptex                                         \
                              : (interr("bad TPTE_SPTR index", ptex, 4), 0)] \
                .aptex)
#define TPTE_VAL(ptex)                                                     \
  APTE_VAL(gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                \
                             ? ptex                                        \
                             : (interr("bad TPTE_VAL index", ptex, 4), 0)] \
               .aptex)
#define TPTE_STRIDE(ptex)                                                \
  APTE_STRIDE(                                                           \
      gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                   \
                        ? ptex                                           \
                        : (interr("bad TPTE_STRIDE index", ptex, 4), 0)] \
          .aptex)
#define TPTE_FLAG(ptex)                                            \
  gpte.stg_base[ptex >= 0 && ptex < gpte.stg_avail                 \
                    ? ptex                                         \
                    : (interr("bad TPTE_FLAG index", ptex, 4), 0)] \
      .flag
#else
#define TPTE_NEXT(ptex) gpte.stg_base[ptex].next
#define TPTE_APTEX(ptex) gpte.stg_base[ptex].aptex
#define TPTE_TYPE(ptex) APTE_TYPE(gpte.stg_base[ptex].aptex)
#define TPTE_SPTR(ptex) APTE_SPTR(gpte.stg_base[ptex].aptex)
#define TPTE_VAL(ptex) APTE_VAL(gpte.stg_base[ptex].aptex)
#define TPTE_STRIDE(ptex) APTE_STRIDE(gpte.stg_base[ptex].aptex)
#endif

/*
 * This is a structure defining pseudo assignments
 * the LHS is a PSD
 * the RHS is a PTE (for direct targets) or PSD (for indirect targets)
 */

typedef struct {
  int type;
  int lhs;
  int rhs;
  int node;
  int iltx;
  int stride;
  int next;
} AS;

static STG_DECLARE(as, AS) = { STG_INIT };

#if DEBUG
#define ASTYPE(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                 \
                  ? asx                                      \
                  : (interr("bad ASTYPE index", asx, 4), 0)] \
      .type
#define ASLHS(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                \
                  ? asx                                     \
                  : (interr("bad ASLHS index", asx, 4), 0)] \
      .lhs
#define ASRHS(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                \
                  ? asx                                     \
                  : (interr("bad ASRHS index", asx, 4), 0)] \
      .rhs
#define ASNODE(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                 \
                  ? asx                                      \
                  : (interr("bad ASNODE index", asx, 4), 0)] \
      .node
#define ASILT(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                \
                  ? asx                                     \
                  : (interr("bad ASILT index", asx, 4), 0)] \
      .iltx
#define ASSTRIDE(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                   \
                  ? asx                                        \
                  : (interr("bad ASSTRIDE index", asx, 4), 0)] \
      .stride
#define ASNEXT(asx)                                          \
  as.stg_base[asx >= 0 && asx < as.stg_avail                 \
                  ? asx                                      \
                  : (interr("bad ASNEXT index", asx, 4), 0)] \
      .next
#else
#define ASTYPE(asx) as.stg_base[asx].type
#define ASLHS(asx) as.stg_base[asx].lhs
#define ASRHS(asx) as.stg_base[asx].rhs
#define ASNODE(asx) as.stg_base[asx].node
#define ASILT(asx) as.stg_base[asx].iltx
#define ASSTRIDE(asx) as.stg_base[asx].stride
#define ASNEXT(asx) as.stg_base[asx].next
#endif

#define AS_DIRECT 1
#define AS_INDIRECT 2
#define AS_CLEAR 3
#define AS_UNK 4
#define AS_UNINIT 5
#define AS_INIT 6
#define AS_ADD_DIRECT 7
#define AS_ADD_INDIRECT 8

#define FIRSTAS(std) STD_PTASGN(std)
static int prevasx;

static STG_DECLARE(head, int) = { STG_INIT };
static int *nodeoffset = NULL;

/* at node n, for slot sl, list of pointer targets starts at HEAD(n,sl) */
#if DEBUG
#define HEAD(v, sl)                                                          \
  head.stg_base[nodeoffset[v >= 0 && v <= opt.num_nodes                      \
                               ? v                                           \
                               : (interr("bad HEAD node index", v, 4), 0)] + \
                (sl >= 0 && sl <= gpsd.nslots                                \
                     ? sl                                                    \
                     : (interr("bad HEAD slot index", sl, 4), 0))]
#else
#define HEAD(v, sl) head.stg_base[nodeoffset[v] + sl]
#endif

static int *localhead = NULL;
#if DEBUG
#define LHEAD(psdx)                          \
  localhead[psdx >= 0 && psdx <= gpsd.nslots \
                ? psdx                       \
                : (interr("bad LHEAD index", psdx, 4), 0)]
#else
#define LHEAD(psdx) localhead[psdx]
#endif

/*
 * keep track of VAL values given to anonymous and global dynamically allocated
 * vars
 */
static STG_DECLARE(ganon, int) = { STG_INIT };

static struct {
  STG_MEMBERS(int);
  int local_dyn;
} gdyn = {STG_INIT, 0};

static struct {
  int tot, ntot;
} gcount = {0, 0};

#if defined(TARGET_WIN_X8664)
static char *importname = "__imp_";
static int importnamelen = 6;
#endif

#if DEBUG
static void
puttds(int tdsx)
{
  switch (TTYPE(tdsx)) {
  case TT_UNK:
    fprintf(gbl.dbgfil, "Unknown");
    break;
  case TT_UNINIT:
    fprintf(gbl.dbgfil, "Uninit");
    break;
  case TT_PSYM:
    fprintf(gbl.dbgfil, "%d:%s", TSYM(tdsx), SYMNAME(TSYM(tdsx)));
    break;
  case TT_ISYM:
    fprintf(gbl.dbgfil, "%d:%s?", TSYM(tdsx), SYMNAME(TSYM(tdsx)));
    break;
  case TT_ANON:
    fprintf(gbl.dbgfil, ".anon%d.", TVAL(tdsx));
    break;
  case TT_GDYN:
    fprintf(gbl.dbgfil, ".dyn%d.", TVAL(tdsx));
    break;
  case TT_IGDYN:
    fprintf(gbl.dbgfil, ".dyn%d?.", TVAL(tdsx));
    break;
  case TT_LDYN:
    fprintf(gbl.dbgfil, ".ldyn%d.", TVAL(tdsx));
    break;
  case TT_ILDYN:
    fprintf(gbl.dbgfil, ".ldyn%d?.", TVAL(tdsx));
    break;
  case TT_CON:
    if (TVAL(tdsx) == 0) {
      fprintf(gbl.dbgfil, ".null.");
    } else {
      fprintf(gbl.dbgfil, ".con.");
    }
    break;
  case TT_IND:
    fprintf(gbl.dbgfil, "*");
    puttds(TVAL(tdsx));
    break;
  case TT_IIND:
    fprintf(gbl.dbgfil, "*(");
    puttds(TVAL(tdsx));
    fprintf(gbl.dbgfil, ")?");
    break;
  case TT_MEM:
    puttds(TVAL(tdsx));
    fprintf(gbl.dbgfil, ".%s", SYMNAME(TSYM(tdsx)));
    break;
  case TT_GLOB:
    /* dummy, used when all globals might be modified */
    fprintf(gbl.dbgfil, ".glob.");
    break;
  case TT_NLOC:
    /* pseudo target for any nonlocal variable */
    fprintf(gbl.dbgfil, ".nloc.");
    break;
  default:
    fprintf(gbl.dbgfil, "?? %d:%d:%d", TTYPE(tdsx), TSYM(tdsx), TVAL(tdsx));
    break;
  }
} /* puttds */

static void
putpsd(int psdx)
{
  switch (PSD_TYPE(psdx)) {
  case TT_PSYM:
    fprintf(gbl.dbgfil, "%d:%s", PSD_SYM(psdx), SYMNAME(PSD_SYM(psdx)));
    break;
  case TT_ISYM:
    fprintf(gbl.dbgfil, "%d:%s?", PSD_SYM(psdx), SYMNAME(PSD_SYM(psdx)));
    break;
  case TT_IND:
    fprintf(gbl.dbgfil, "*(");
    putpsd(PSD_PARENT(psdx));
    fprintf(gbl.dbgfil, ")");
    break;
  case TT_IIND:
    fprintf(gbl.dbgfil, "*(");
    putpsd(PSD_PARENT(psdx));
    fprintf(gbl.dbgfil, ")?");
    break;
  case TT_MEM:
    putpsd(PSD_PARENT(psdx));
    fprintf(gbl.dbgfil, ".%s", SYMNAME(PSD_SYM(psdx)));
    break;
  case TT_GLOB:
    fprintf(gbl.dbgfil, ".glob.");
    break;
  case TT_NLOC:
    fprintf(gbl.dbgfil, ".nloc!.");
    break;
  case TT_UNK:
    fprintf(gbl.dbgfil, "Unknown!");
    break;
  case TT_UNINIT:
    fprintf(gbl.dbgfil, "Uninit!");
    break;
  default:
    fprintf(gbl.dbgfil, ".??%d:%d.", PSD_TYPE(psdx), PSD_SYM(psdx));
    break;
  }
} /* putpsd */

static void
putpte(int ptex)
{
  fprintf(gbl.dbgfil, "[%d:%d]", ptex, TPTE_APTEX(ptex));
  switch (TTMASK(TPTE_TYPE(ptex))) {
  case TT_UNINIT:
    fprintf(gbl.dbgfil, "&.uninit.");
    break;
  case TT_UNK:
    fprintf(gbl.dbgfil, "&.unk.");
    break;
  case TT_PSYM:
    fprintf(gbl.dbgfil, "&%d:%s", TPTE_SPTR(ptex), SYMNAME(TPTE_SPTR(ptex)));
    break;
  case TT_ISYM:
    fprintf(gbl.dbgfil, "&%d:%s?", TPTE_SPTR(ptex), SYMNAME(TPTE_SPTR(ptex)));
    break;
  case TT_ANON:
    fprintf(gbl.dbgfil, "&anon:%d", TPTE_VAL(ptex));
    break;
  case TT_GDYN:
    fprintf(gbl.dbgfil, "&dyn:%d", TPTE_VAL(ptex));
    break;
  case TT_IGDYN:
    fprintf(gbl.dbgfil, "&dyn:%d?", TPTE_VAL(ptex));
    break;
  case TT_LDYN:
    fprintf(gbl.dbgfil, "&ldyn:%d", TPTE_VAL(ptex));
    break;
  case TT_ILDYN:
    fprintf(gbl.dbgfil, "&ldyn:%d?", TPTE_VAL(ptex));
    break;
  case TT_CON:
    fprintf(gbl.dbgfil, "&con:%d", TPTE_VAL(ptex));
    break;
  case TT_NLOC:
    fprintf(gbl.dbgfil, "&.nloc.");
    break;
  case TT_IND:
    /* should not appear */
    fprintf(gbl.dbgfil, "*(");
    putpte(TPTE_VAL(ptex));
    fprintf(gbl.dbgfil, ")");
    break;
  case TT_IIND:
    /* should not appear */
    fprintf(gbl.dbgfil, "*(");
    putpte(TPTE_VAL(ptex));
    fprintf(gbl.dbgfil, ")?");
    break;
  case TT_MEM:
    /* should not appear */
    putpte(TPTE_VAL(ptex));
    fprintf(gbl.dbgfil, ".%s", SYMNAME(TPTE_SPTR(ptex)));
    break;
  case TT_GLOB:
    /* should not appear */
    fprintf(gbl.dbgfil, ".glob%d!.", TPTE_VAL(ptex));
    break;
  default:
    /* should not appear */
    fprintf(gbl.dbgfil, "???%d:%d", TPTE_TYPE(ptex), TPTE_VAL(ptex));
    break;
  }
  if (TPTE_STRIDE(ptex) > 0) {
    fprintf(gbl.dbgfil, "(::%d)", TPTE_STRIDE(ptex));
  }
  if (TTMUST(TPTE_TYPE(ptex))) {
    fprintf(gbl.dbgfil, "!");
  }
} /* putpte */

static void
putassign(int asx)
{
  fprintf(gbl.dbgfil, "          as:%d node:%d ", asx, ASNODE(asx));
  fprintf(gbl.dbgfil, "std:%d  ", ASILT(asx));
  putpsd(ASLHS(asx));
  switch (ASTYPE(asx)) {
  case AS_DIRECT:
    fprintf(gbl.dbgfil, "  <=  ");
    putpte(ASRHS(asx));
    break;
  case AS_ADD_DIRECT:
    fprintf(gbl.dbgfil, "  <+=  ");
    putpte(ASRHS(asx));
    break;
  case AS_INDIRECT:
    fprintf(gbl.dbgfil, "  <==  ");
    putpsd(ASRHS(asx));
    break;
  case AS_ADD_INDIRECT:
    fprintf(gbl.dbgfil, "  <+==  ");
    putpsd(ASRHS(asx));
    break;
  case AS_INIT:
    fprintf(gbl.dbgfil, "  <=  ");
    putpte(ASRHS(asx));
    fprintf(gbl.dbgfil, "  (init)");
    break;
  case AS_CLEAR:
    fprintf(gbl.dbgfil, "  <==  CLEAR");
    break;
  case AS_UNK:
    fprintf(gbl.dbgfil, "  <==  UNKNOWN");
    break;
  default:
    fprintf(gbl.dbgfil, "  <<-- type:%d(%d)", ASTYPE(asx), ASRHS(asx));
    break;
  }
  fprintf(gbl.dbgfil, "   stride(%d)", ASSTRIDE(asx));
  fprintf(gbl.dbgfil, "\n");
} /* putassign */

void
putstdassigns(int stdx)
{
  int a;
  if (as.stg_base) {
    for (a = FIRSTAS(stdx); a > 0; a = ASNEXT(a)) {
      putassign(a);
    }
  }
} /* putstdassigns */

static void
putptelist(int ptelistx)
{
  int ptex;
  for (ptex = ptelistx; ptex > 0; ptex = TPTE_NEXT(ptex)) {
    putpte(ptex);
    if (TPTE_NEXT(ptex) > 0) {
      fprintf(gbl.dbgfil, ", ");
    }
  }
} /* putptelist */

static void
puttarget(int sl)
{
  int psdx;
  psdx = SLOT(sl);
  fprintf(gbl.dbgfil, "psd:%d ", psdx);
  putpsd(psdx);
  if (LHEAD(sl) == TTE_NULL) {
    fprintf(gbl.dbgfil, " --> ...");
  } else if (LHEAD(sl) == TPTE_UNK) {
    fprintf(gbl.dbgfil, " --> unknown");
  } else {
    fprintf(gbl.dbgfil, " --> ");
    putptelist(LHEAD(sl));
  }
  fprintf(gbl.dbgfil, "\n");
} /* puttarget */

static void
puttargets(void)
{
  int sl;
  for (sl = 1; sl < gpsd.nslots; ++sl) {
    puttarget(sl);
  }
} /* puttargets */

static void
putnodetarget(const char *ch, int v, int sl)
{
  int psdx;
  psdx = SLOT(sl);
  fprintf(gbl.dbgfil, "%s node:%d psd:%d ", ch, v, psdx);
  putpsd(psdx);
  if (HEAD(v, sl) == TTE_NULL) {
    fprintf(gbl.dbgfil, " --> ...");
  } else if (HEAD(v, sl) == TPTE_UNK) {
    fprintf(gbl.dbgfil, " --> unknown");
  } else {
    fprintf(gbl.dbgfil, " --> ");
    putptelist(HEAD(v, sl));
  }
  fprintf(gbl.dbgfil, "\n");
} /* puttarget */

static void
putnodetargets(const char *ch, int v)
{
  int sl;

  for (sl = 1; sl < gpsd.nslots; ++sl) {
    putnodetarget(ch, v, sl);
  }
} /* puttargets */
#endif

/*
 * return TRUE if this source is a global symbol,
 * or is an indirection that might point to a global symbol.
 */
static int
is_source_global(int psdx)
{
  int sptr, parentpsdx, sl, ptex;
  switch (PSD_TYPE(psdx)) {
  case TT_PSYM:
  case TT_ISYM:
    /* is this a global symbol? */
    sptr = PSD_SYM(psdx);
    if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK) {
      return TRUE;
    }
    if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
      sptr = MIDNUMG(sptr);
      if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK) {
        return TRUE;
      }
    }
    break;
  case TT_IND:
  case TT_IIND:
    /* look at parent */
    /* look at pointer targets of the parent */
    parentpsdx = PSD_PARENT(psdx);
    if (is_source_global(parentpsdx))
      return TRUE;
    sl = PSD_SLOT(parentpsdx);
    if (LHEAD(sl) == TPTE_UNK)
      return TRUE;
    for (ptex = LHEAD(sl); ptex > 0; ptex = TPTE_NEXT(ptex)) {
      switch (TTMASK(TPTE_TYPE(ptex))) {
      case TT_UNK:
      case TT_UNINIT:
        return TRUE;
      case TT_PSYM:
      case TT_ISYM:
        sptr = TPTE_SPTR(ptex);
        if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK) {
          return TRUE;
        }
        if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
          sptr = MIDNUMG(sptr);
          if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK) {
            return TRUE;
          }
        }
        break;
      case TT_NLOC:
        return TRUE; /* all we know is something nonlocal, may be global */
      case TT_ANON:
      case TT_GDYN:
      case TT_IGDYN:
      case TT_LDYN:
      case TT_ILDYN:
      case TT_CON:
        break;
      default:
        /* real error */
        interr("bad PTE type", TPTE_TYPE(ptex), ERR_Fatal);
        return TRUE;
      }
    }
    return FALSE;
    break;
  case TT_GLOB:
    return TRUE;
  case TT_UNK:
  case TT_UNINIT:
    return TRUE;
  }
  return FALSE;
} /* is_source_global */

/*
 * return TRUE if this source is a global symbol that we know to be safe
 */
static int
safe_symbol(int psdx)
{
  int sptr;
  switch (PSD_TYPE(psdx)) {
  case TT_PSYM:
  case TT_ISYM:
    /* is this a global symbol? */
    sptr = PSD_SYM(psdx);
    if (VOLG(sptr))
      return FALSE;
    break;
    /* we don't know anything about other pointer sources */
  }
  return FALSE;
} /* safe_symbol */

/*
 * return TRUE if this source is a global or static symbol,
 * or is an indirection that might point to a global or static symbol.
 */
static int
is_source_nonlocal(int psdx)
{
  int sptr, parentpsdx, sl, ptex;
  switch (PSD_TYPE(psdx)) {
  case TT_PSYM:
  case TT_ISYM:
    /* is this a global symbol? */
    sptr = PSD_SYM(psdx);
    if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK ||
        SCG(sptr) == SC_DUMMY || (gbl.internal > 1 && !INTERNALG(sptr))) {
      return TRUE;
    }
    if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
      sptr = MIDNUMG(sptr);
      if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK ||
          SCG(sptr) == SC_DUMMY || (gbl.internal > 1 && !INTERNALG(sptr))) {
        return TRUE;
      }
    }
    break;
  case TT_IND:
  case TT_IIND:
    /* look at pointer targets of the parent */
    parentpsdx = PSD_PARENT(psdx);
    if (is_source_nonlocal(parentpsdx))
      return TRUE;
    sl = PSD_SLOT(parentpsdx);
    if (LHEAD(sl) == TPTE_UNK)
      return TRUE;
    for (ptex = LHEAD(sl); ptex > 0; ptex = TPTE_NEXT(ptex)) {
      switch (TTMASK(TPTE_TYPE(ptex))) {
      case TT_UNK:
      case TT_UNINIT:
        return TRUE;
      case TT_PSYM:
      case TT_ISYM:
        sptr = TPTE_SPTR(ptex);
        if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK ||
            SCG(sptr) == SC_DUMMY || (gbl.internal > 1 && !INTERNALG(sptr))) {
          return TRUE;
        }
        if (SCG(sptr) == SC_BASED && MIDNUMG(sptr)) {
          sptr = MIDNUMG(sptr);
          if (SCG(sptr) == SC_EXTERN || SCG(sptr) == SC_CMBLK ||
              SCG(sptr) == SC_DUMMY || (gbl.internal > 1 && !INTERNALG(sptr))) {
            return TRUE;
          }
        }
        break;
      case TT_NLOC: /* all we know is something nonlocal, may be global */
      case TT_ANON: /* anonymous nonlocal */
      case TT_GLOB:
        return TRUE;
      case TT_GDYN:
      case TT_IGDYN:
      case TT_LDYN:
      case TT_ILDYN:
      case TT_CON:
        break;
      default:
        /* real error */
        interr("bad PTE type", TPTE_TYPE(ptex), ERR_Warning);
        return TRUE;
      }
    }
    return FALSE;
    break;
  case TT_GLOB:
    return TRUE;
  case TT_UNINIT:
  case TT_UNK:
    return TRUE;
  }
  return FALSE;
} /* is_source_nonlocal */

/*
 * add a new pointer source descriptor to the table
 * if this is an LHS reference, make sure it has a slot number
 */
static void
add_psd(int tpdx, int type, int parent, int sym, bool lhs)
{
  int val, psdx, t;

  t = TTMASK(type);
  val = (int)((t ^ sym ^ parent) & 0x7fff) % PSDHSZ;
  for (psdx = psdhsh[val]; psdx > 0; psdx = PSD_HASHLK(psdx)) {
    if (PSD_TYPE(psdx) == t && PSD_PARENT(psdx) == parent &&
        PSD_SYM(psdx) == sym) {
      if (tpdx)
        TLINK(tpdx) = psdx;
      if (TTIMP(type)) {
        PSD_TYPE(psdx) |= TTIMPFLAG;
      }
      if (lhs && PSD_SLOT(psdx) == 0) {
        PSD_SLOT(psdx) = gpsd.nslots++;
      }
      return;
    }
  }
  psdx = STG_NEXT(gpsd);
  PSD_TYPE(psdx) = type;
  PSD_PARENT(psdx) = parent;
  PSD_SYM(psdx) = sym;
  PSD_NEXT(psdx) = 0;
  PSD_CHILD(psdx) = 0;
  PSD_SLOT(psdx) = 0;
  PSD_HASHLK(psdx) = psdhsh[val];
  psdhsh[val] = psdx;
  if (tpdx)
    TLINK(tpdx) = psdx;
  if (parent) {
    PSD_NEXT(psdx) = PSD_CHILD(parent);
    PSD_CHILD(parent) = psdx;
  }
  if (lhs) {
    PSD_SLOT(psdx) = gpsd.nslots++;
  } else if (type == TT_PSYM || type == TT_ISYM) {
    if (is_source_nonlocal(psdx)) {
      PSD_SLOT(psdx) = gpsd.nslots++;
    }
  }
} /* add_psd */

/*
 * call add_psd to build a pointer source descriptor for this entry;
 * make sure all dependent (parent) pointer source descriptors are also built.
 */
static void
build_psd(int tdsx, bool lhs)
{
  int subtds;
  if (TLINK(tdsx))
    return;
  switch (TTYPE(tdsx)) {
  case TT_PSYM:
    add_psd(tdsx, TT_PSYM, 0, TSYM(tdsx), lhs);
    break;
  case TT_ISYM:
    add_psd(tdsx, TT_ISYM, 0, TSYM(tdsx), lhs);
    break;
  case TT_IND:
    subtds = TVAL(tdsx);
    build_psd(subtds, lhs);
    add_psd(tdsx, TT_IND, TLINK(subtds), 0, lhs);
    break;
  case TT_IIND:
    subtds = TVAL(tdsx);
    build_psd(subtds, lhs);
    add_psd(tdsx, TT_IIND, TLINK(subtds), 0, lhs);
    break;
  case TT_MEM:
    subtds = TVAL(tdsx);
    build_psd(subtds, lhs);
    add_psd(tdsx, TT_MEM, TLINK(subtds), TSYM(tdsx), lhs);
    break;
  case TT_GLOB:
    add_psd(tdsx, TT_GLOB, 0, 0, lhs);
    break;
  default:
    TLINK(tdsx) = 0;
    break;
  }
} /* build_psd */

/*
 * find pointer source descriptor in the table
 */
static int
find_psd(int type, int parent, int sym)
{
  int val, psdx;

  val = (int)((type ^ sym ^ parent) & 0x7fff) % PSDHSZ;
  for (psdx = psdhsh[val]; psdx > 0; psdx = PSD_HASHLK(psdx)) {
    if (PSD_TYPE(psdx) == type && PSD_PARENT(psdx) == parent &&
        PSD_SYM(psdx) == sym) {
      return psdx;
    }
  }
  return TTE_NULL;
} /* find_psd */

/*
 * get APTE entry, lookup in hash table
 */
static int
get_apte(int type, int sptr, int val, int stride)
{
  int h, aptex;
  h = (type * 51 + sptr + val) << 4;
  h = h & 0xffffff;
  h = h % APTEHSZ;
  for (aptex = aptehsh[h]; aptex; aptex = APTE_HASHLK(aptex)) {
    if (APTE_TYPE(aptex) == type && APTE_SPTR(aptex) == sptr &&
        APTE_VAL(aptex) == val
        && APTE_STRIDE(aptex) == stride
        ) {
      return aptex;
    }
  }
  aptex = STG_NEXT(apte);
  APTE_TYPE(aptex) = type;
  APTE_SPTR(aptex) = sptr;
  APTE_VAL(aptex) = val;
  APTE_MARK(aptex) = 0;
  APTE_STRIDE(aptex) = stride;
  APTE_HASHLK(aptex) = aptehsh[h];
  aptehsh[h] = aptex;
  return aptex;
} /* get_apte */

/*
 * find but don't add APTE entry, lookup in hash table
 */
static int
find_apte(int type, int sptr, int val)
{
  int h, aptex;
  h = (type * 51 + sptr + val) << 4;
  h = h & 0xffffff;
  h = h % APTEHSZ;
  for (aptex = aptehsh[h]; aptex; aptex = APTE_HASHLK(aptex)) {
    if (APTE_TYPE(aptex) == type && APTE_SPTR(aptex) == sptr &&
        APTE_VAL(aptex) == val) {
      return aptex;
    }
  }
  return -1;
} /* find_apte */

/*
 * get new pte entry, use free list if available
 */
static int
get_pte(int type, int sptr, int val, int stride)
{
  int ptex;
  ptex = gpte.xstg_free;
  if (ptex > 0) {
    gpte.xstg_free = TPTE_NEXT(ptex);
  } else {
    ptex = STG_NEXT(gpte);
  }
  TPTE_APTEX(ptex) = get_apte(type, sptr, val, stride);
  TPTE_NEXT(ptex) = TTE_NULL;
  return ptex;
} /* get_pte */

/*
 * return the imprecise version of this ptetype
 */
static int
imprecise_ptetype(int ptetype)
{
  if (ptetype == TT_PSYM)
    ptetype = TT_ISYM;
  else if (ptetype == TT_GDYN)
    ptetype = TT_IGDYN;
  else if (ptetype == TT_LDYN)
    ptetype = TT_ILDYN;
  else if (ptetype == TT_IND)
    ptetype = TT_IIND;
  return ptetype;
} /* imprecise_ptetype */

/*
 * copy pte tree
 */
static int
copy_pte(int ptex, int imprecise, int mstride)
{
  int newptex, ptetype;
  ptetype = TTMASK(TPTE_TYPE(ptex));
  if (imprecise)
    ptetype = imprecise_ptetype(ptetype);
  newptex = get_pte(ptetype, TPTE_SPTR(ptex), TPTE_VAL(ptex),
                    TPTE_STRIDE(ptex) * mstride
                    );
  switch (TTMASK(TPTE_TYPE(newptex))) {
  case TT_IND:
  case TT_IIND:
  case TT_MEM:
    TPTE_VAL(newptex) = copy_pte(TPTE_VAL(ptex), imprecise, 0);
    break;
  }
  return newptex;
} /* copy_pte */

/*
 * free a single pte
 */
static void
free_pte(int ptex)
{

  switch (TTMASK(TPTE_TYPE(ptex))) {
  case TT_IND:
  case TT_IIND:
  case TT_MEM:
    free_pte(TPTE_VAL(ptex));
    break;
  }
  TPTE_APTEX(ptex) = 0;
  TPTE_NEXT(ptex) = gpte.xstg_free;
  gpte.xstg_free = ptex;
} /* free_pte */

/*
 * clear the TPTE list at LHEAD(sl)
 */
static void
free_list(int ptelistx)
{
  int ptex, nextptex;
  nextptex = ptelistx;
  for (ptex = nextptex; ptex > 0; ptex = nextptex) {
    nextptex = TPTE_NEXT(ptex);
    free_pte(ptex);
  }
} /* free_list */

/*
 * clear the TPTE list at LHEAD(sl)
 */
static void
free_list_slot(int sl)
{
  free_list(LHEAD(sl));
  LHEAD(sl) = TTE_NULL;
} /* free_list_slot */

/*
 * reverse depth first ordering, and rdfo number for each node
 */
static int rdfocount, *rdfo, *rdfonum;
#if DEBUG
#define RDFO(r) \
  rdfo[r >= 0 && r <= opt.num_nodes ? r : (interr("bad RDFO index", r, 4), 0)]
#define RDFONUM(r)                     \
  rdfonum[r >= 0 && r <= opt.num_nodes \
              ? r                      \
              : (interr("bad RDFONUM index", r, 4), 0)]
#else
#define RDFO(r) rdfo[r]
#define RDFONUM(r) rdfonum[r]
#endif

/*
 * v is a flow graph number; build reverse-depth-first order
 * of flow graph nodes
 */
static void
buildrdfo(int v)
{
  PSI_P s;
  int sv;
  RDFONUM(v) = -1;
  for (s = FG_SUCC(v); s != PSI_P_NULL; s = PSI_NEXT(s)) {
    sv = PSI_NODE(s);
    if (RDFONUM(sv) == 0) {
      buildrdfo(sv);
    }
  }
  ++rdfocount;
  RDFO(rdfocount) = v;
  RDFONUM(v) = rdfocount;
} /* buildrdfo */

static int
make_address(int astx, int pt, int sym, int offset)
{
  int tdsx;
  if (pt == TT_UNK)
    return 0;
  tdsx = STG_NEXT(gtpd);
  TTYPE(tdsx) = pt;
  TSYM(tdsx) = sym;
  TVAL(tdsx) = offset;
  return tdsx;
} /* make_address */

/*
 * fill in temporary pointer structure descriptors
 */
static int
make_address_for_ast(int astx, int parentast)
{
  int a, sptr, ttype, ret;
  ttype = TT_PSYM;
  a = astx;
  sptr = 0;
  while (!sptr) {
    switch (A_TYPEG(a)) {
    case A_ID:
      sptr = A_SPTRG(a);
      break;
    case A_MEM:
      if (parentast && A_TKNG(parentast) == TK_ALLOCATE &&
          A_SPTRG(A_MEMG(a)) && DTY(DTYPEG(A_SPTRG(A_MEMG(a)))) == TY_ARRAY) {
        /* process allocate of an array that is a structure member */
        sptr = A_SPTRG(A_MEMG(a));
      }
      a = A_PARENTG(a);
      ttype = TT_ISYM;
      break;
    case A_SUBSCR:
      a = A_LOPG(a);
      ttype = TT_ISYM;
      break;
    default:
      sptr = -1;
      break;
    }
  }
  if (sptr > 0) {
    ret = make_address(astx, ttype, sptr, 0);
  } else {
    ret = make_address(astx, TT_UNK, 0, 0);
  }
  return ret;
} /* make_address_for_ast */

/*
 * get ptr source descriptor of LHS
 * decide if RHS is a real pointer target or indirect through pointer source
 * create pseudo assignment to represent all this
 * stride is only used for F90
 */
static void
make_assignment(int v, int iltx, int lhstds, int rhstds, int add, int stride)
{
  int lhspsdx, asx, rhsptex, rhspsdx;
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT)) {
    fprintf(gbl.dbgfil, "node:%d bih:%d ilt:%d lhs:%d <= rhs:%d  ", v,
            FG_TO_BIH(v), iltx, lhstds, rhstds);
    puttds(lhstds);
    fprintf(gbl.dbgfil, " <= ");
    puttds(rhstds);
    fprintf(gbl.dbgfil, "\n");
    printast(iltx);
    fprintf(gbl.dbgfil, " (stride=%d)\n", stride);
  }
#endif
  build_psd(lhstds, true);
  lhspsdx = TLINK(lhstds);
  asx = 0;
  switch (TTYPE(rhstds)) {
  case TT_PSYM:
  case TT_ISYM:
    /* real pointer target */
    rhsptex = get_pte(TTYPE(rhstds), TSYM(rhstds), 0,
                      stride
                      );

    asx = STG_NEXT(as);
    ASTYPE(asx) = add ? AS_ADD_DIRECT : AS_DIRECT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = rhsptex;
    ASNODE(asx) = v;
    ASILT(asx) = iltx;
    ASNEXT(asx) = 0;
    ASSTRIDE(asx) = stride;
    break;

  case TT_CON:
  case TT_GDYN:
  case TT_IGDYN:
  case TT_LDYN:
  case TT_ILDYN:
    /* real pointer target */
    rhsptex = get_pte(TTYPE(rhstds), TSYM(rhstds), TVAL(rhstds),
                      stride
                      );

    asx = STG_NEXT(as);
    ASTYPE(asx) = add ? AS_ADD_DIRECT : AS_DIRECT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = rhsptex;
    ASNODE(asx) = v;
    ASILT(asx) = iltx;
    ASNEXT(asx) = 0;
    ASSTRIDE(asx) = stride;
    break;
  case TT_UNK:
    asx = STG_NEXT(as);
    ASTYPE(asx) = AS_UNK;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = 0;
    ASNODE(asx) = v;
    ASILT(asx) = iltx;
    ASNEXT(asx) = 0;
    ASSTRIDE(asx) = stride;
    break;
  case TT_UNINIT:
    asx = STG_NEXT(as);
    ASTYPE(asx) = AS_UNINIT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = 0;
    ASNODE(asx) = v;
    ASILT(asx) = iltx;
    ASNEXT(asx) = 0;
    ASSTRIDE(asx) = stride;
    break;
  default:
    /* indirect or otherwise; need dataflow analysis */
    build_psd(rhstds, false);
    rhspsdx = TLINK(rhstds);

    asx = STG_NEXT(as);
    ASTYPE(asx) = add ? AS_ADD_INDIRECT : AS_INDIRECT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = rhspsdx;
    ASNODE(asx) = v;
    ASILT(asx) = iltx;
    ASNEXT(asx) = 0;
    ASSTRIDE(asx) = stride;
    break;
  }
  if (asx) {
    if (prevasx) {
      ASNEXT(prevasx) = asx;
    } else {
      FIRSTAS(v) = asx;
    }
    prevasx = asx;
  }
#if DEBUG
  if (asx && DBGBIT(TRACEFLAG, TRACEBIT)) {
    putassign(asx);
    fprintf(gbl.dbgfil, "\n");
  }
#endif
} /* make_assignment */

/*
 * Find the stride of the leftmost dimension of this pointer target reference
 * Return zero if it's not a constant.
 */
static int
find_stride(int astx)
{
  if (A_TYPEG(astx) == A_ID) {
    /* pointer points to whole array */
    return 1;
  } else if (A_TYPEG(astx) == A_SUBSCR) {
    /* find stride of leftmost subscript */
    int asd, astss, sptr;
    asd = A_ASDG(astx);
    astss = ASD_SUBS(asd, 0);
    if (A_TYPEG(astss) != A_TRIPLE) {
      return 0;
    }
    astss = A_STRIDEG(astss);
    if (astss == 0)
      return 1;
    if (!A_ALIASG(astss))
      return 0;
    astss = A_ALIASG(astss);
    if (A_TYPEG(astss) != A_CNST)
      return 0;
    sptr = A_SPTRG(astss);
    switch (DTY(DTYPEG(sptr))) {
    case TY_INT8:
    case TY_INT:
    case TY_SINT:
    case TY_BINT:
      if (CONVAL1G(sptr) != 0)
        return 0;
      return CONVAL2G(sptr);
    default:
      return 0;
    }
  } else {
    return 0;
  }
} /* find_stride */

/*
 * find pointer assignments
 * characterize both LHS and RHS as pointer source descriptors
 * this must also handle other assignments that change pointers
 * and treat them as unknown assignments
 * as well as function calls with pointer arguments
 */
static int globalv;

static void
_find_pointer_assignments_f90(int astx, int *pstdx)
{
  int lop, allglobals, allargs, dpdsc, funcsptr;
  int dummy, args, argcnt, a, arg, argsptr;
  switch (A_TYPEG(astx)) {
  case A_ICALL:
    /* intrinsic call, see if it is ptr assignment */
    if (A_OPTYPEG(astx) == I_PTR2_ASSIGN) {
      /* pointer assignment */
      int args, lhsastx, rhsastx, lhsdsx, rhsdsx, stride;
      args = A_ARGSG(astx);
      lhsastx = ARGT_ARG(args, 0);
      rhsastx = ARGT_ARG(args, 1);
      lhsdsx = make_address_for_ast(lhsastx, astx);
      rhsdsx = make_address_for_ast(rhsastx, astx);
      stride = find_stride(rhsastx);
      make_assignment(globalv, *pstdx, lhsdsx, rhsdsx, 0, stride);
    } else if (A_OPTYPEG(astx) == I_NULLIFY) {
      /* pointer nullify */
      int args, lhsastx, lhsdsx, rhsdsx;
      args = A_ARGSG(astx);
      lhsastx = ARGT_ARG(args, 0);
      lhsdsx = make_address_for_ast(lhsastx, astx);
      rhsdsx = make_address(astx, TT_CON, 0, 0);
      make_assignment(globalv, *pstdx, lhsdsx, rhsdsx, 0, 0);
    }
    break;
  case A_CALL:
    lop = A_LOPG(astx);
    allglobals = 1;
    allargs = 0;
    if (A_TYPEG(lop) == A_ID) {
      funcsptr = A_SPTRG(lop);
      dpdsc = DPDSCG(funcsptr);
      if (DPDSCG(funcsptr)) {
        allargs = 1; /* all pointer dummy args might be modified */
      }
    }
    if (allglobals) {
      make_assignment(globalv, *pstdx, 2, 0, 0, 0);
    }
    if (allargs) {
      /* have funcsptr, dpdsc */
      args = A_ARGSG(astx);
      argcnt = A_ARGCNTG(astx);
      for (a = 0; a < argcnt; ++a) {
        arg = ARGT_ARG(args, a);
        if (A_TYPEG(arg) == A_ID) {
          argsptr = A_SPTRG(arg);
          if (POINTERG(argsptr)) {
            int lhsdsx, rhsdsx;
            /* if the pointer might otherwise be modified */
            dummy = aux.dpdsc_base[dpdsc + a];
            if (dummy && POINTERG(dummy)) {
              /* yes, it might */
              lhsdsx = make_address_for_ast(arg, astx);
              rhsdsx = make_address(astx, TT_UNK, 0, 0);
              make_assignment(globalv, *pstdx, lhsdsx, rhsdsx, 0, 0);
            }
          }
        }
      }
    }
    break;
  case A_ALLOC:
    /* allocate/deallocate */
    if (A_TKNG(astx) == TK_ALLOCATE) {
      /* allocate */
      int objastx, lhsdsx, rhsdsx;
      objastx = A_SRCG(astx);
      if (A_TYPEG(objastx) == A_SUBSCR) {
        objastx = A_LOPG(objastx);
      }
      lhsdsx = make_address_for_ast(objastx, astx);
      rhsdsx = make_address(astx, TT_LDYN, 0, gdyn.local_dyn++);
      make_assignment(globalv, *pstdx, lhsdsx, rhsdsx, 0, 1);
    } else {
      /* deallocate */
      int objastx, lhsdsx, rhsdsx;
      objastx = A_SRCG(astx);
      if (A_TYPEG(objastx) == A_SUBSCR) {
        objastx = A_LOPG(objastx);
      }
      lhsdsx = make_address_for_ast(objastx, astx);
      rhsdsx = make_address(astx, TT_CON, 0, 0);
      make_assignment(globalv, *pstdx, lhsdsx, rhsdsx, 0, 0);
    }
    break;
  }
} /* _find_pointer_assignments_f90 */

/*
 * symbolically interpret the statements in this statement
 * represent the pointer target information as a TPTE list
 */
static void
find_pointer_assignments_f90(int stdx)
{
  prevasx = 0;
  gtpd.stg_avail = 3;
  ast_visit(1, 1);
  globalv = stdx;
  ast_traverse(STD_AST(stdx), NULL, _find_pointer_assignments_f90, &stdx);
  ast_unvisit();
} /* find_pointer_assignments_f90 */

#ifdef FLANG_POINTSTO_UNUSED
/*
 * return unique identifier for an anonymous variable
 */
static int
anonymous_number(int n)
{
  int a;
  for (a = 1; a < ganon.stg_avail; ++a) {
    if (ganon.stg_base[a] == n)
      return a;
  }
  a = STG_NEXT(ganon);
  ganon.stg_base[a] = n;
  return a;
} /* anonymous_number */
#endif

#ifdef FLANG_POINTSTO_UNUSED
/*
 * return unique identifier for a dynamically-allocated block of space
 */
static int
dynamic_number(int n)
{
  int d;
  for (d = 1; d < gdyn.stg_avail; ++d) {
    if (gdyn.stg_base[d] == n)
      return d;
  }
  d = STG_NEXT(gdyn);
  gdyn.stg_base[d] = n;
  return d;
} /* dynamic_number */
#endif

#ifdef FLANG_POINTSTO_UNUSED
/*
 * Add pseudo assignments for initial pointer information at the program entry
 */
static void
make_init_assignment(int v, int sourcesptr, int stars, int targettype,
                     int targetinfo)
{
  int lhspsdx, asx, rhsptex;
  int lhstds;
  /* get LHS tds for source */
  Trace(("init assignment node %d sourcesptr %d stars=%d targettype=%d "
         "targetinfo=%d",
         v, sourcesptr, stars, targettype, targetinfo));
  lhstds = make_address(0, TT_PSYM, sourcesptr, 0);
  while (stars-- > 0) {
    lhstds = make_address(0, TT_IND, 0, lhstds);
  }
  build_psd(lhstds, true);
  lhspsdx = TLINK(lhstds);
  asx = 0;
  /* get PTE for RHS */
  switch (targettype) {
  case 1:
    /* another SPTR */
    rhsptex = get_pte(TT_ISYM, targetinfo, 0, 0);
    asx = STG_NEXT(as);
    ASTYPE(asx) = AS_INIT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = rhsptex;
    ASNODE(asx) = v;
    ASILT(asx) = 0;
    ASNEXT(asx) = 0;
    break;
  case 2:
    /* anonymous variable */
    rhsptex = get_pte(TT_ANON, 0, anonymous_number(targetinfo), 0);
    asx = STG_NEXT(as);
    ASTYPE(asx) = AS_INIT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = rhsptex;
    ASNODE(asx) = v;
    ASILT(asx) = 0;
    ASNEXT(asx) = 0;
    break;
  case 3:
    /* allocatable storage */
    rhsptex = get_pte(TT_GDYN, 0, dynamic_number(targetinfo), 0);
    asx = STG_NEXT(as);
    ASTYPE(asx) = AS_INIT;
    ASLHS(asx) = lhspsdx;
    ASRHS(asx) = rhsptex;
    ASNODE(asx) = v;
    ASILT(asx) = 0;
    ASNEXT(asx) = 0;
    break;
  default:
    interr("bad target type", targettype, ERR_Fatal);
    return;
  }
  if (asx) {
    ASNEXT(asx) = FIRSTAS(v);
    FIRSTAS(v) = asx;
  }
} /* make_init_assignment */
#endif

/*
 * add the TPTE at ptex to the TPTE list at LHEAD(s)
 */
static void
add_list(int ptex, int psdx, int imprecise, int mstride)
{
  int newptex;
  newptex = copy_pte(ptex, imprecise, mstride);
  TPTE_NEXT(newptex) = LHEAD(PSD_SLOT(psdx));
  LHEAD(PSD_SLOT(psdx)) = newptex;
} /* add_list */

/*
 * copy the TPTE list at T to the TPTE list at position LHEAD(s)
 */
static int
copy_list(int ptelistx)
{
  int ptex, newptex, headptex;
  if (ptelistx == TPTE_UNK) {
    return TPTE_UNK;
  } else {
    headptex = TTE_NULL;
    for (ptex = ptelistx; ptex > 0; ptex = TPTE_NEXT(ptex)) {
      newptex = copy_pte(ptex, 0, 1);
      TPTE_NEXT(newptex) = headptex;
      headptex = newptex;
    }
    return headptex;
  }
} /* copy_list */

/*
 * we're at a point where we know nothing, assume worst case
 */
static void
unk_all(void)
{
  int sl;
  for (sl = 1; sl < gpsd.nslots; ++sl) {
    free_list_slot(sl);
    LHEAD(sl) = TPTE_UNK;
  }
} /* unk_all */

/*
 * clear all information for globals, or anything that might point to a global
 */
static void
unk_globals(void)
{
  int sl;
  for (sl = 1; sl < gpsd.nslots; ++sl) {
    if (is_source_global(SLOT(sl)) && !safe_symbol(SLOT(sl))) {
      free_list_slot(sl);
      LHEAD(sl) = TPTE_UNK;
    }
  }
} /* unk_globals */

/*
 * set up the APTE flags
 * mark any APTE that might be a target of the given psd
 *  we're looking for the case
 *   p = &q;
 *   q = &a;
 *   so we have to add &a to the target list of *p as well.
 */
static unsigned int marker = 0;
static unsigned int unknown_mark = 0, nonlocal_mark = 0, some_nonlocal_mark = 0;
static void
mark_apte_targets(int psdx)
{
  int parentpsdx, parentsl, ptex, sptr, aptex;
  ++marker;
  switch (PSD_TYPE(psdx)) {
  case TT_PSYM:
  case TT_ISYM:
    aptex = find_apte(TT_PSYM, PSD_SYM(psdx), 0);
    if (aptex > 0)
      APTE_MARK(aptex) = marker;
    aptex = find_apte(TT_ISYM, PSD_SYM(psdx), 0);
    if (aptex > 0)
      APTE_MARK(aptex) = marker;
    sptr = PSD_SYM(psdx);
    break;
  case TT_IND:
  case TT_IIND:
    parentpsdx = PSD_PARENT(psdx);
    /* mark targets of parentpsdx */
    parentsl = PSD_SLOT(parentpsdx);
    if (LHEAD(parentsl) == TPTE_UNK) {
      unknown_mark = marker;
    } else {
      for (ptex = LHEAD(parentsl); ptex > 0; ptex = TPTE_NEXT(ptex)) {
        switch (TTMASK(TPTE_TYPE(ptex))) {
        case TT_UNK:
        case TT_UNINIT:
          unknown_mark = marker;
          some_nonlocal_mark = marker;
          break;
        case TT_PSYM:
        case TT_ISYM:
          sptr = TPTE_SPTR(ptex);
          aptex = find_apte(TT_PSYM, sptr, 0);
          if (aptex > 0)
            APTE_MARK(aptex) = marker;
          aptex = find_apte(TT_ISYM, sptr, 0);
          if (aptex > 0)
            APTE_MARK(aptex) = marker;
          break;
        case TT_NLOC: /* all we know is something nonlocal, may be global */
          nonlocal_mark = marker;
          some_nonlocal_mark = marker;
          break;
        case TT_ANON: /* anonymous nonlocal */
        case TT_GDYN:
        case TT_IGDYN:
        case TT_LDYN:
        case TT_ILDYN:
          some_nonlocal_mark = marker;
          break;
        case TT_CON:
          aptex = find_apte(TTMASK(TPTE_TYPE(ptex)), 0, TPTE_VAL(ptex));
          if (aptex > 0)
            APTE_MARK(aptex) = marker;
          break;
        default:
          /* real error */
          interr("bad PTE type", TPTE_TYPE(ptex), ERR_Warning);
          break;
        }
      }
    }
    break;
  case TT_MEM:
    /* ### */
    break;
  case TT_GLOB:
  case TT_NLOC:
  case TT_UNK:
  case TT_UNINIT:
  default:
    interr("bad PSD type", PSD_TYPE(psdx), ERR_Fatal);
    break;
  }
} /* mark_apte_targets */

/*
 * return TRUE if the pointer source at psdx might
 * intersect with the target at targetpsdx, that is
 * psdx is an indirect with a parent PSD that might point to targetpsdx
 */
static int
might_target(int psdx)
{
  int parentpsdx, parentsl, ptex, sptr, aptex;
  switch (PSD_TYPE(psdx)) {
  case TT_IND:
  case TT_IIND:
    /* see if the parent might point to the lhs;
     * changing the lhs changes the parent pointee, which is this */
    parentpsdx = PSD_PARENT(psdx);
    parentsl = PSD_SLOT(parentpsdx);
    if (LHEAD(parentsl) == TPTE_UNK)
      return TRUE;
    if (unknown_mark == marker)
      return TRUE;
    for (ptex = LHEAD(parentsl); ptex > 0; ptex = TPTE_NEXT(ptex)) {
      switch (TTMASK(TPTE_TYPE(ptex))) {
      case TT_UNK:
      case TT_UNINIT:
        return TRUE;
      case TT_PSYM:
      case TT_ISYM:
        aptex = TPTE_APTEX(ptex);
        if (APTE_MARK(aptex) == marker)
          return TRUE;
        break;
      case TT_NLOC: /* all we know is something nonlocal, may be global */
        if (some_nonlocal_mark == marker)
          return TRUE;
        break;
      case TT_ANON: /* anonymous nonlocal */
      case TT_GDYN:
      case TT_IGDYN:
      case TT_LDYN:
      case TT_ILDYN:
      case TT_CON:
        aptex = TPTE_APTEX(ptex);
        if (APTE_MARK(aptex) == marker)
          return TRUE;
        break;
      default:
        /* real error */
        interr("bad PTE type", TPTE_TYPE(ptex), ERR_Warning);
        return TRUE;
      }
    }
    break;
  case TT_ISYM:
  case TT_PSYM:
    aptex = find_apte(TT_PSYM, PSD_SYM(psdx), 0);
    if (aptex > 0 && APTE_MARK(aptex) == marker)
      return TRUE;
    aptex = find_apte(TT_ISYM, PSD_SYM(psdx), 0);
    if (aptex > 0 && APTE_MARK(aptex) == marker)
      return TRUE;
    sptr = PSD_SYM(psdx);
    break;
    /* ignore the other types, which don't / can't point to the LHS */
  }
  return FALSE;
} /* might_target */

static int CHANGES;

/*
 * Check all the pte's in the list for the lhs
 * if the lhs target list is unknown, stop here
 * if the pte is 'unknown', change lhs target list to unknown, increment
 * CHANGES, stop
 * if there is a match, stop here, no changes
 * if there is a less precise one match, stop here, no changes
 * if there is a more precise one, make it less precise, increment CHANGES, stop
 * here
 * otherwise, add this pte to the list, increment CHANGES
 *  stride is only used for F90
 */
static void
_addtpte(int lhspsdx, int ptex, int imprecise, int mstride)
{
  int ptetype, lpte, lptetype, ptes, ptestype, lptes, lptestype, match, sl;
  int rhspsdx, rsl;
  sl = PSD_SLOT(lhspsdx);
  if (LHEAD(sl) == TPTE_UNK)
    return;
  if (ptex == TPTE_UNK) {
    ++CHANGES;
    free_list_slot(sl);
    LHEAD(sl) = TPTE_UNK;
    return;
  }
  ptetype = TTMASK(TPTE_TYPE(ptex));
  if (imprecise)
    ptetype = imprecise_ptetype(ptetype);
  /* in PGF90, we can't have a pointer to a pointer.
   * p => b where b is a pointer or allocatable means make p
   * point to what b points to */
  if ((ptetype == TT_PSYM || ptetype == TT_ISYM) && TPTE_SPTR(ptex) &&
      (POINTERG(TPTE_SPTR(ptex)) || ALLOCATTRG(TPTE_SPTR(ptex)))) {
    /* get the RHS PSD entry */
    rhspsdx = find_psd(TT_PSYM, 0, TPTE_SPTR(ptex));
    /* if p => a, and a is a pointer or allocatable, then p can point to
     * anything that a can point to */
    if (rhspsdx <= 0) {
      _addtpte(lhspsdx, TPTE_UNK, 0, 0);
    } else {
      int rptex, imp;
      rsl = PSD_SLOT(rhspsdx);
      if (rsl) {
        imp = 0;
        if (ptetype == TT_ISYM || imprecise_ptetype(ptetype) == ptetype)
          imp = 1;
        for (rptex = LHEAD(rsl); rptex > 0; rptex = TPTE_NEXT(rptex)) {
          _addtpte(lhspsdx, rptex, imp, TPTE_STRIDE(ptex));
        }
      }
    }
    return;
  }
  match = 0;
  for (lpte = LHEAD(sl); lpte > 0; lpte = TPTE_NEXT(lpte)) {
    lptetype = TTMASK(TPTE_TYPE(lpte));
    if (imprecise)
      lptetype = imprecise_ptetype(lptetype);
    if (lptetype == ptetype && TPTE_SPTR(lpte) == TPTE_SPTR(ptex) &&
        TPTE_VAL(lpte) == TPTE_VAL(ptex)
        && TPTE_STRIDE(lpte) * mstride == TPTE_STRIDE(lpte)
            ) {
      return;
    }
    /* see if this is indirection from the same symbol */
    match = 1;
    for (lptes = lpte, ptes = ptex; lptes > 0 && ptes > 0;
         lptes = TPTE_VAL(lptes), ptes = TPTE_VAL(ptes)) {
      lptestype = TTMASK(TPTE_TYPE(lptes));
      if (imprecise)
        lptestype = imprecise_ptetype(lptestype);
      ptestype = TTMASK(TPTE_TYPE(ptes));
      if (TPTE_STRIDE(lpte) * mstride != TPTE_STRIDE(lpte)) {
        match = 0;
        break;
      }
      if (lptestype == TT_IND || lptestype == TT_IIND) {
        if (ptestype != TT_IND && ptestype != TT_IIND) {
          match = 0;
          break;
        } else if (TPTE_SPTR(ptes) != TPTE_SPTR(lptes)) {
          match = 0;
          break;
        }
      } else if (lptestype == TT_PSYM || lptestype == TT_ISYM) {
        if (ptestype != TT_PSYM && ptestype != TT_ISYM) {
          match = 0;
          break;
        } else if (TPTE_SPTR(ptes) != TPTE_SPTR(lptes)) {
          match = 0;
          break;
        }
      } else if (lptestype == TT_GDYN || lptestype == TT_IGDYN) {
        if (ptestype != TT_GDYN && ptestype != TT_IGDYN) {
          match = 0;
          break;
        } else if (TPTE_VAL(ptes) != TPTE_VAL(lptes)) {
          match = 0;
          break;
        }
        break;
      } else if (lptestype == TT_LDYN || lptestype == TT_ILDYN) {
        if (ptestype != TT_LDYN && ptestype != TT_ILDYN) {
          match = 0;
          break;
        } else if (TPTE_VAL(ptes) != TPTE_VAL(lptes)) {
          match = 0;
          break;
        }
        break;
      } else {
        match = 0;
        break;
      }
    }
    if (match) {
      /* change lptes to the more imprecise of lptes and ptex, and be done with
       * it */
      for (lptes = lpte, ptes = ptex; lptes > 0 && ptes > 0;
           lptes = TPTE_VAL(lptes), ptes = TPTE_VAL(ptes)) {
        lptestype = TTMASK(TPTE_TYPE(lptes));
        if (imprecise)
          lptestype = imprecise_ptetype(lptestype);
        ptestype = TTMASK(TPTE_TYPE(ptes));
        if (ptestype == TT_IIND) {
          if (lptestype == TT_IND) {
            TPTE_APTEX(lptes) =
                get_apte(TT_IIND, TPTE_SPTR(lptes), TPTE_VAL(lptes),
                         TPTE_STRIDE(lptes) * mstride
                         );
            ++CHANGES;
          }
          /* loop */
        } else if (ptestype == TT_IND) {
          /* loop */
        } else if (ptestype == TT_ISYM) {
          if (lptestype == TT_PSYM) {
            TPTE_APTEX(lptes) =
                get_apte(TT_ISYM, TPTE_SPTR(lptes), TPTE_VAL(lptes),
                         TPTE_STRIDE(lptes) * mstride
                         );
            ++CHANGES;
          }
          break;
        } else if (ptestype == TT_PSYM) {
          break;
        } else if (ptestype == TT_IGDYN) {
          if (lptestype == TT_GDYN) {
            TPTE_APTEX(lptes) =
                get_apte(TT_IGDYN, TPTE_SPTR(lptes), TPTE_VAL(lptes),
                         TPTE_STRIDE(lptes) * mstride
                         );
            ++CHANGES;
          }
          break;
        } else if (ptestype == TT_GDYN) {
          break;
        } else if (ptestype == TT_ILDYN) {
          if (lptestype == TT_LDYN) {
            TPTE_APTEX(lptes) =
                get_apte(TT_ILDYN, TPTE_SPTR(lptes), TPTE_VAL(lptes),
                         TPTE_STRIDE(lptes) * mstride
                         );
            ++CHANGES;
          }
          break;
        } else if (ptestype == TT_LDYN) {
          break;
        } else {
          break;
        }
      }
    }
  }
  if (!match) {
    /* no match found, add */
    ++CHANGES;
    add_list(ptex, lhspsdx, imprecise, mstride);
  }

} /* _addtpte */

static void
addtpte(int lhspsdx, int ptex)
{
  int sl;
  _addtpte(lhspsdx, ptex, 0, 1);
  /* pointers that might target lhs must be modified as well */
  mark_apte_targets(lhspsdx);
  for (sl = 1; sl < gpsd.nslots; ++sl) {
    if (SLOT(sl) != lhspsdx && might_target(SLOT(sl))) {
      _addtpte(SLOT(sl), ptex, 1, 0);
    }
  }
} /* addtpte */

static void
inittpte(int lhspsdx, int ptex)
{
  _addtpte(lhspsdx, ptex, 0, 1);
} /* inittpte */

/*
 * remove the target information for the LHS
 * add the single target from the RHS
 * must modify *LHS to match *RHS
 * anything that can point to LHS must also be modified
 */
static void
replacepte(int lhspsdx, int ptex)
{
  free_list_slot(PSD_SLOT(lhspsdx));
  addtpte(lhspsdx, ptex);
} /* replacepte */

static void
add_tptelist(int lhspsdx, int ptelistx)
{
  int ptex, sl;
  if (ptelistx == TPTE_UNK) {
    _addtpte(lhspsdx, ptelistx, 1, 1);
    /* pointers that might target lhs must be modified as well */
    mark_apte_targets(lhspsdx);
    for (sl = 1; sl < gpsd.nslots; ++sl) {
      if (SLOT(sl) != lhspsdx && might_target(SLOT(sl))) {
        _addtpte(SLOT(sl), ptelistx, 1, 1);
      }
    }
  } else {
    for (ptex = ptelistx; ptex > 0; ptex = TPTE_NEXT(ptex)) {
      _addtpte(lhspsdx, ptex, 0, 1);
    }
    /* pointers that might target lhs must be modified as well */
    /* This is the expensive part:
     *
     *    int** p1, *p2, *p3, *p4, **p5;
     *    int a, b;
     *
     *    p1 = test1 ? &p3 : &p4;	-- here, p1 => p3|p4
     *    p2 = test2 ? &a : &b;		-- here, p2 => a|b
     *
     *    *p1 = p2;
     *
     * We know p1 might point to p3 or p4 and p2 might point to a or b,
     * so we must propagage target a|b to p3 and p4 as well as **p1
     *
     * If we add the assignment
     *    p5 = test3 ? &p3 : NULL;
     * before the *p1 assignment, then p5 might point to p3
     * we must then propagate the target a|b to *p5 as well
     *
     * So, when assigning a pointer value to *p1, any pointer target of p1
     * has its pointer targets updated by the pointer targets of the RHS.
     *
     * In fact, in any pointer assignment, any pointer that might alias
     * the LHS also has its pointer targets updated
     */
    mark_apte_targets(lhspsdx);
    for (sl = 1; sl < gpsd.nslots; ++sl) {
      if (SLOT(sl) != lhspsdx && might_target(SLOT(sl))) {
        for (ptex = ptelistx; ptex > 0; ptex = TPTE_NEXT(ptex)) {
          _addtpte(SLOT(sl), ptex, 1, 1);
        }
      }
    }
  }
} /* add_tptelist */

/*
 *  replace target information for LHS by the list
 */
static void
replace_tptelist(int lhspsdx, int ptelistx)
{
  free_list_slot(PSD_SLOT(lhspsdx));
  add_tptelist(lhspsdx, ptelistx);
} /* replace_tptelist */

/*
 * look at pointer targets of the LHS, add RHS targets to anything
 * that the LHS might point to
 */
static void
add_target_tptelist(int lhspsdx, int ptelistx)
{
  int sl;
  add_tptelist(lhspsdx, ptelistx);
  sl = PSD_SLOT(lhspsdx);
  if (LHEAD(sl) == TPTE_UNK) {
    /* everything changes */
    unk_all();
  }
} /* add_target_tptelist */

/*
 * take the RHS of the assignment, compute the effective TPTE list
 */
static int
effective_rhs(int psdx)
{
  int parentpsdx, sl, ptelistx, ptex;
  switch (PSD_TYPE(psdx)) {
  case TT_UNINIT:
    return TPTE_UNINIT;

  case TT_UNK:
    /* totally unknown */
    return TPTE_UNK;

  case TT_IND:
    /* return the list of the target */
    parentpsdx = PSD_PARENT(psdx);
    sl = PSD_SLOT(parentpsdx);
    if (sl == 0) {
      return TPTE_UNK;
    }
    return copy_list(LHEAD(sl));
  case TT_IIND:
    /* find the list of the target */
    parentpsdx = PSD_PARENT(psdx);
    sl = PSD_SLOT(parentpsdx);
    if (sl == 0) {
      return TPTE_UNK;
    }
    ptelistx = copy_list(LHEAD(sl));
    /* now account for the imprecision */
    for (ptex = ptelistx; ptex > 0; ptex = TPTE_NEXT(ptex)) {
      switch (TTMASK(TPTE_TYPE(ptex))) {
      case TT_PSYM:
        /* make imprecise */
        TPTE_APTEX(ptex) = get_apte(TT_ISYM, TPTE_SPTR(ptex), TPTE_VAL(ptex),
                                    TPTE_STRIDE(ptex)
                                        );
        break;
      case TT_ISYM:
        /* already imprecise */
        break;
      case TT_IND:
        /* make imprecise */
        TPTE_APTEX(ptex) = get_apte(TT_IIND, TPTE_SPTR(ptex), TPTE_VAL(ptex),
                                    TPTE_STRIDE(ptex)
                                        );
        break;
      case TT_IIND:
        /* already imprecise */
        break;
      case TT_UNK:
      case TT_UNINIT:
        free_list(ptelistx);
        return TPTE_UNK;
      case TT_CON:
        free_list(ptelistx);
        return TPTE_UNK;
      }
    }
    return ptelistx;
  default:
    /* real error */
    interr("pointsto: unknown RHS type in assignment", PSD_TYPE(psdx), ERR_Fatal);
    return TPTE_UNK;
  }
} /* effective_rhs */

/*
 * symbolically interpret the pointer assignments in this flow graph node
 * represent the pointer target information as a TPTE list
 */
static void
interpret(int asx)
{
  int lhspsdx, rhsptelistx, sl, stride;
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT)) {
    putassign(asx);
  }
#endif
  stride = ASSTRIDE(asx);
  /* interpret assignment as changes to pointer target information */
  /* pointer equals address is easy, reset pointer target */
  /* pointer equals fetch(pointer) is easy, copy pointer targets */
  /* pointer equals fetch(pointer)imprecise is easy, copy pointer targets,
   *  set imprecise */
  /* pointer equals unknown is easy, clear information */
  /* glob equals unknown is easy, clear information for all globals,
   * anything that might alias a global */
  switch (ASTYPE(asx)) {
  case AS_DIRECT:
    /* RHS must be precise or imprecise symbol. */
    lhspsdx = ASLHS(asx);
    switch (PSD_TYPE(lhspsdx)) {
    case TT_PSYM:
      /* LHS is precise symbol; this is a MUST POINTS TO relationship */
      replacepte(lhspsdx, ASRHS(asx));
      break;

    case TT_ISYM:
      /* LHS is imprecise symbol; this is a MAY POINTS TO relationship */
      addtpte(lhspsdx, ASRHS(asx));
      break;

    case TT_IND:
      /* LHS is indirect; MUST POINTS TO if all indirects are precise */
      replacepte(lhspsdx, ASRHS(asx));
      break;
    case TT_IIND:
      /* LHS is imprecise indirect; MAY POINTS TO */
      addtpte(lhspsdx, ASRHS(asx));
      break;
    case TT_GLOB:
      /* happens at calls.  Mark all globals as unknown */
      unk_globals();
      break;
    case TT_UNK:
    case TT_UNINIT: /* should not happen */
      unk_all();
      break;
    case TT_MEM:
      /* not used yet */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Warning);
      unk_all();
      break;
    default:
      /* real error */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Fatal);
      unk_all();
      break;
    }
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      puttarget(PSD_SLOT(lhspsdx));
    }
#endif
    break;
  case AS_ADD_DIRECT:
    /* RHS must be precise or imprecise symbol. */
    lhspsdx = ASLHS(asx);
    switch (PSD_TYPE(lhspsdx)) {
    case TT_PSYM:
    case TT_ISYM:
    case TT_IND:
    case TT_IIND:
      /* LHS is imprecise indirect; MAY POINTS TO */
      addtpte(lhspsdx, ASRHS(asx));
      break;
    case TT_GLOB:
      /* happens at calls.  Mark all globals as unknown */
      unk_globals();
      break;
    case TT_UNK:
    case TT_UNINIT: /* should not happen */
      unk_all();
      break;
    case TT_MEM:
      /* not used yet */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Warning);
      unk_all();
      break;
    default:
      /* real error */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Fatal);
      unk_all();
      break;
    }
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      puttarget(PSD_SLOT(lhspsdx));
    }
#endif
    break;
  case AS_INIT:
    /* RHS must be precise or imprecise symbol. */
    lhspsdx = ASLHS(asx);
    sl = PSD_SLOT(lhspsdx);
    if (LHEAD(sl) == TPTE_UNK) {
      free_list_slot(sl);
    }
    switch (PSD_TYPE(lhspsdx)) {
    case TT_PSYM:
      /* LHS is precise symbol; this is a MUST POINTS TO relationship */
      inittpte(lhspsdx, ASRHS(asx));
      break;

    case TT_ISYM:
      /* LHS is imprecise symbol; this is a MAY POINTS TO relationship */
      inittpte(lhspsdx, ASRHS(asx));
      break;

    case TT_IND:
      /* LHS is indirect; MUST POINTS TO if all indirects are precise */
      inittpte(lhspsdx, ASRHS(asx));
      break;
    case TT_IIND:
      /* LHS is imprecise indirect; MAY POINTS TO */
      inittpte(lhspsdx, ASRHS(asx));
      break;
    case TT_UNK:
    case TT_UNINIT: /* should not happen */
      unk_all();
      break;
    case TT_MEM:
      /* not used yet */
      interr("pointsto: unknown LHS type in init assignment", PSD_TYPE(lhspsdx),
             2);
      unk_all();
      break;
    default:
      /* real error */
      interr("pointsto: unknown LHS type in init assignment", PSD_TYPE(lhspsdx),
             4);
      break;
    }
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      puttarget(PSD_SLOT(lhspsdx));
    }
#endif
    break;
  case AS_INDIRECT:
    /* Get the effect of the RHS */
    rhsptelistx = effective_rhs(ASRHS(asx));
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      fprintf(gbl.dbgfil, " effective target --> ");
      putptelist(rhsptelistx);
      fprintf(gbl.dbgfil, "\n");
    }
#endif
    lhspsdx = ASLHS(asx);
    switch (PSD_TYPE(lhspsdx)) {
    case TT_PSYM:
      /* LHS is precise symbol; this is a MUST POINTS TO relationship */
      replace_tptelist(lhspsdx, rhsptelistx);
      break;

    case TT_ISYM:
      /* LHS is imprecise symbol; this is a MAY POINTS TO relationship */
      add_tptelist(lhspsdx, rhsptelistx);
      break;

    case TT_GLOB:
      /* this is used when globals might be modified */
      replace_tptelist(lhspsdx, rhsptelistx);
      /* must also modify target list for any global pointer source */
      for (sl = 1; sl < gpsd.nslots; ++sl) {
        if (is_source_global(SLOT(sl))) {
          add_tptelist(SLOT(sl), rhsptelistx);
        }
      }
      break;

    case TT_IND:
    case TT_IIND:
      /* need to handle this */
      add_target_tptelist(lhspsdx, rhsptelistx);
      Trace(("indirect LHS type in assignment"));
      break;
    case TT_UNK:
    case TT_UNINIT: /* should not happen */
      unk_all();
      break;
    case TT_MEM:
      /* ### really want to handle member LHS types */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Warning);
      Trace(("unknown LHS type in assignment"));
      break;
    default:
      /* real error */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Fatal);
      Trace(("unknown LHS type in assignment"));
      break;
    }
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      puttarget(PSD_SLOT(lhspsdx));
    }
#endif
    free_list(rhsptelistx);
    break;
  case AS_ADD_INDIRECT:
    /* Get the effect of the RHS */
    rhsptelistx = effective_rhs(ASRHS(asx));
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      fprintf(gbl.dbgfil, " effective target --> ");
      putptelist(rhsptelistx);
      fprintf(gbl.dbgfil, "\n");
    }
#endif
    lhspsdx = ASLHS(asx);
    switch (PSD_TYPE(lhspsdx)) {
    case TT_PSYM:
    case TT_ISYM:
      /* LHS is imprecise symbol; this is a MAY POINTS TO relationship */
      add_tptelist(lhspsdx, rhsptelistx);
      break;

    case TT_GLOB:
      /* this is used when globals might be modified */
      add_tptelist(lhspsdx, rhsptelistx);
      /* must also modify target list for any global pointer source */
      for (sl = 1; sl < gpsd.nslots; ++sl) {
        if (is_source_global(SLOT(sl))) {
          add_tptelist(SLOT(sl), rhsptelistx);
        }
      }
      break;

    case TT_IND:
    case TT_IIND:
      /* need to handle this */
      add_target_tptelist(lhspsdx, rhsptelistx);
      Trace(("indirect LHS type in assignment"));
      break;
    case TT_UNK:
    case TT_UNINIT: /* should not happen */
      unk_all();
      break;
    case TT_MEM:
      /* ### really want to handle member LHS types */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Warning);
      Trace(("unknown LHS type in assignment"));
      break;
    default:
      /* real error */
      interr("pointsto: unknown LHS type in assignment", PSD_TYPE(lhspsdx), ERR_Fatal);
      Trace(("unknown LHS type in assignment"));
      break;
    }
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      puttarget(PSD_SLOT(lhspsdx));
    }
#endif
    free_list(rhsptelistx);
    break;
  case AS_UNK:
    lhspsdx = ASLHS(asx);
    switch (PSD_TYPE(lhspsdx)) {
    case TT_GLOB:
      /* shouldn't happen.  Mark all globals as unknown */
      unk_globals();
      break;
    case TT_UNK:
    case TT_UNINIT: /* should not happen */
      unk_all();
      break;
    default:
      replace_tptelist(lhspsdx, TPTE_UNK);
      break;
    }
    break;
  case AS_CLEAR:
    break;
  default:
    break;
  }
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT)) {
    Trace(("after assignment"));
    puttargets();
  }
#endif
} /* interpret */

/*
 * compare lptex and ptex; if they have the same or comparable type,
 * compare subtypes.  If they all compare exactly, return ptex.
 * if they compare imprecisely, change ptex to be as imprecise as lptex,
 * and return -ptex if changes were made.
 */
static int
imprecise_match(int lptex, int ptex)
{
  int m;
  if (TPTE_STRIDE(lptex) != TPTE_STRIDE(ptex))
    return 0;
  if (TTMASK(TPTE_TYPE(lptex)) == TTMASK(TPTE_TYPE(ptex))) {
    switch (TTMASK(TPTE_TYPE(lptex))) {
    case TT_PSYM:
    case TT_ISYM:
      if (TPTE_SPTR(lptex) == TPTE_SPTR(ptex))
        return ptex;
      return 0;

    case TT_IND:
    case TT_IIND:
      return imprecise_match(TPTE_VAL(lptex), TPTE_VAL(ptex));

    case TT_GDYN:
    case TT_IGDYN:
    case TT_LDYN:
    case TT_ILDYN:
    case TT_ANON:
      if (TPTE_VAL(lptex) == TPTE_VAL(ptex))
        return ptex;
      return 0;

    case TT_CON:
      if (TPTE_VAL(lptex) == TPTE_VAL(ptex))
        return ptex;
      return 0;
    case TT_UNINIT:
    case TT_UNK:
    case TT_NLOC:
      return ptex;
    default:
      /* real error */
      interr("pointsto: unknown TPTE target type", TPTE_TYPE(lptex), ERR_Fatal);
      return 0;
    }
  } else {
    /* not quite an exact match */
    switch (TTMASK(TPTE_TYPE(lptex))) {
    case TT_PSYM:
      /* might match TT_ISYM, if the same symbol */
      if (TTMASK(TPTE_TYPE(ptex)) == TT_ISYM &&
          TPTE_SPTR(lptex) == TPTE_SPTR(ptex))
        return ptex;
      return 0;
    case TT_ISYM:
      /* might match TT_PSYM, if the same symbol, but change to ISYM */
      if (TTMASK(TPTE_TYPE(ptex)) == TT_PSYM &&
          TPTE_SPTR(lptex) == TPTE_SPTR(ptex)) {
        TPTE_APTEX(ptex) = get_apte(TT_ISYM, TPTE_SPTR(ptex), TPTE_VAL(ptex),
                                    TPTE_STRIDE(ptex)
                                        );
        return -ptex;
      }
      return 0;
    case TT_IND:
      /* might match TT_IIND, if the subtypes match */
      if (TTMASK(TPTE_TYPE(ptex)) == TT_IIND) {
        m = imprecise_match(TPTE_VAL(lptex), TPTE_VAL(ptex));
        if (m > 0)
          return ptex;
        if (m < 0)
          return -ptex;
      }
      return 0;
    case TT_IIND:
      /* might match TT_IND, if the subtypes match, but change to IIND */
      if (TTMASK(TPTE_TYPE(ptex)) == TT_IND) {
        m = imprecise_match(TPTE_VAL(lptex), TPTE_VAL(ptex));
        if (m != 0) {
          TPTE_APTEX(ptex) = get_apte(TT_IIND, TPTE_SPTR(ptex), TPTE_VAL(ptex),
                                      TPTE_STRIDE(ptex)
                                          );
          return -ptex;
        }
      }
      return 0;
    default:
      /* no other imprecise matches */
      return 0;
    }
  }
} /* imprecise_match */

/*
 * look along the list at ptelistx; find a match for lptex;
 * an imprecise match is good enough.  If lptex is less precise
 * than the one found, change the one found to be just as imprecise.
 * return 0 if not found, >0 if exact match found, <0 if changes were made
 */
static int
find_imprecise_match(int lptex, int ptelistx)
{
  int ptex, m;
  for (ptex = ptelistx; ptex > 0; ptex = TPTE_NEXT(ptex)) {
    m = imprecise_match(lptex, ptex);
    if (m != 0)
      return m;
  }
  return 0;
} /* find_imprecise_match */

/*
 * combine (take the 'meet') of the information at the head of
 * node v with the new local information we have.  return TRUE if it
 * changes at all
 */
static int
pte_changed(int v)
{
  int changes = 0;
  int sl;
  int lptex, hptex, newptex, newptelistx;
  Trace(("Combine information into node %d block %d", v, FG_TO_BIH(v)));
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT))
    putnodetargets("before", v);
#endif
  for (sl = 1; sl < gpsd.nslots; ++sl) {
    if (HEAD(v, sl) == TPTE_UNK) {
      /* already worst case, give up */
    } else if (LHEAD(sl) == TPTE_UNK) {
      free_list(HEAD(v, sl));
      HEAD(v, sl) = TPTE_UNK;
      ++changes;
    } else if (HEAD(v, sl) == TTE_NULL || HEAD(v, sl) == TPTE_UNINIT) {
      if (LHEAD(sl) != TTE_NULL) {
        HEAD(v, sl) = copy_list(LHEAD(sl));
        ++changes;
      }
    } else if (LHEAD(sl) == TTE_NULL) {
      /* nothing to add */
    } else {
      /* compare pointer targets one at a time */
      /* collect all new PTEs here */
      newptelistx = TTE_NULL;
      for (lptex = LHEAD(sl); lptex > 0; lptex = TPTE_NEXT(lptex)) {
        /* find a matching or almost-matching TPTE in HEAD */
        hptex = find_imprecise_match(lptex, HEAD(v, sl));
        if (hptex == 0) {
          newptex = copy_pte(lptex, 0, 1);
          TPTE_NEXT(newptex) = newptelistx;
          newptelistx = newptex;
          ++changes;
        } else if (hptex < 0) {
          /* flag that something changed */
          ++changes;
        }
      }
      /* add newptelistx to HEAD(v,sl) */
      if (newptelistx > 0) {
        for (newptex = newptelistx; TPTE_NEXT(newptex) > 0;
             newptex = TPTE_NEXT(newptex))
          ;
        TPTE_NEXT(newptex) = HEAD(v, sl);
        HEAD(v, sl) = newptelistx;
      }
    }
  }
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT) && changes)
    putnodetargets(" after", v);
#endif
  return changes;
} /* pte_changed */

#if DEBUG
static int
check_this_assignment(int asx, int v)
{
  int ptex;
  if (ASTYPE(asx) == AS_DIRECT || ASTYPE(asx) == AS_INIT ||
      ASTYPE(asx) == AS_ADD_DIRECT) {
    ptex = ASRHS(asx);
    if (ptex >= gpte.stg_avail) {
      fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, assignment %d, but "
                          "past avail point %d\n",
              ptex, v, asx, gpte.stg_avail);
      return 1;
    } else if (TPTE_FLAG(ptex) == 0) {
      TPTE_FLAG(ptex) = -asx;
    } else if (TPTE_FLAG(ptex) == -9999999) {
      fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, assignment %d, but "
                          "also on free list\n",
              ptex, v, asx);
      return 1;
    } else if (TPTE_FLAG(ptex) < 0) {
      fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, assignment %d, but "
                          "also on list for assignment %d\n",
              ptex, v, asx, -TPTE_FLAG(ptex));
      return 1;
    } else if (TPTE_FLAG(ptex) > 0) {
      fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, assignment %d, but "
                          "also on list for node %d\n",
              ptex, v, asx, TPTE_FLAG(ptex));
      return 1;
    }
  }
  return 0;
} /* check_this_assignment */

/*
 * see that all TPTE entries are on a list somewhere
 */
static void
check_pte(const char *ch)
{
  int v, sl, ptex, bad, asx, stdx;
  for (ptex = 1; ptex < gpte.stg_avail; ++ptex)
    TPTE_FLAG(ptex) = 0;
  /* mark the free list */
  bad = 0;
  for (ptex = gpte.xstg_free; ptex > 0; ptex = TPTE_NEXT(ptex)) {
    if (ptex >= gpte.stg_avail) {
      fprintf(gbl.dbgfil, "TPTE(%d) is on free list but past avail point %d\n",
              ptex, gpte.stg_avail);
      bad = 1;
      break;
    } else if (TPTE_FLAG(ptex) == 0) {
      TPTE_FLAG(ptex) = -9999999;
    } else {
      fprintf(gbl.dbgfil, "TPTE(%d) is on free list twice\n", ptex);
      bad = 1;
      break;
    }
  }
  /* mark pte on target list of each slot at each node */
  for (v = 1; v <= opt.num_nodes; ++v) {
    if (nodeoffset[v] == 0)
      continue;
    for (sl = 1; sl < gpsd.nslots; ++sl) {
      if (HEAD(v, sl) != TPTE_UNINIT) {
        for (ptex = HEAD(v, sl); ptex > 0; ptex = TPTE_NEXT(ptex)) {
          if (ptex >= gpte.stg_avail) {
            fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, slot %d, but "
                                "past avail point %d\n",
                    ptex, v, sl, gpte.stg_avail);
            bad = 1;
            break;
          } else if (TPTE_FLAG(ptex) == 0) {
            TPTE_FLAG(ptex) = v;
          } else if (TPTE_FLAG(ptex) == -9999999) {
            fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, slot %d, but "
                                "also on free list\n",
                    ptex, v, sl);
            bad = 1;
            break;
          } else if (TPTE_FLAG(ptex) < 0) {
            fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, slot %d, but "
                                "also on list for assignment %d\n",
                    ptex, v, sl, -TPTE_FLAG(ptex));
            bad = 1;
            break;
          } else if (TPTE_FLAG(ptex) > 0) {
            fprintf(gbl.dbgfil, "TPTE(%d) is on list for node %d, slot %d, but "
                                "also on list for node %d\n",
                    ptex, v, sl, TPTE_FLAG(ptex));
            bad = 1;
            break;
          }
        }
      }
    }
/* mark pte on direct assignments */
    for (stdx = FG_STDFIRST(v); stdx; stdx = STD_NEXT(stdx)) {
      for (asx = FIRSTAS(stdx); asx > 0; asx = ASNEXT(asx)) {
        bad += check_this_assignment(asx, stdx);
      }
      if (stdx == FG_STDLAST(v))
        break;
    }
    /* also check any unreachable statements following this node up to
     * the next node */
    stdx = FG_STDLAST(v);
    for (stdx = STD_NEXT(stdx); stdx && STD_FG(stdx) == 0;
         stdx = STD_NEXT(stdx)) {
      for (asx = FIRSTAS(stdx); asx > 0; asx = ASNEXT(asx)) {
        bad += check_this_assignment(asx, stdx);
      }
    }
  }
  for (ptex = 2; ptex < gpte.stg_avail; ++ptex) {
    if (TPTE_FLAG(ptex) == 0) {
      fprintf(gbl.dbgfil, "TPTE(%d) is not on any list\n", ptex);
      bad = 1;
    }
  }
  if (bad) {
    /* real error, consistency check failure */
    interr(ch, 0, ERR_Fatal);
  }
} /* check_pte */
#endif

                   /*
                    * save the information as a list of aliases for each statement
                    * head of list at STD_PTA
                    */
typedef struct {
  int next; /* next pointer target */
  int type; /* pointer target type */
  int sptr; /* type-specific value */
  int val;  /* type-specific value */
  int stride;
} FPTE;

static STG_DECLARE(fpte, FPTE);

typedef struct {
  int next;    /* next pointer source */
  int type;    /* type of source */
  int sptr;    /* symbol */
  int ptelist; /* list of pointer targets */
} FPSRC;

static STG_DECLARE(fpsrc, FPSRC);

#if DEBUG
#define FPTE_NEXT(n)                                            \
  fpte.stg_base[n >= 0 && n < fpte.stg_avail                    \
                    ? n                                         \
                    : (interr("bad FPTE_NEXT index", n, 4), 0)] \
      .next
#define FPTE_TYPE(n)                                            \
  fpte.stg_base[n >= 0 && n < fpte.stg_avail                    \
                    ? n                                         \
                    : (interr("bad FPTE_TYPE index", n, 4), 0)] \
      .type
#define FPTE_SPTR(n)                                            \
  fpte.stg_base[n >= 0 && n < fpte.stg_avail                    \
                    ? n                                         \
                    : (interr("bad FPTE_SPTR index", n, 4), 0)] \
      .sptr
#define FPTE_VAL(n)                                            \
  fpte.stg_base[n >= 0 && n < fpte.stg_avail                   \
                    ? n                                        \
                    : (interr("bad FPTE_VAL index", n, 4), 0)] \
      .val
#define FPTE_STRIDE(n)                                            \
  fpte.stg_base[n >= 0 && n < fpte.stg_avail                      \
                    ? n                                           \
                    : (interr("bad FPTE_STRIDE index", n, 4), 0)] \
      .stride

#define FPSRC_NEXT(n)                                             \
  fpsrc.stg_base[n >= 0 && n < fpsrc.stg_avail                    \
                     ? n                                          \
                     : (interr("bad FPSRC_NEXT index", n, 4), 0)] \
      .next
#define FPSRC_TYPE(n)                                             \
  fpsrc.stg_base[n >= 0 && n < fpsrc.stg_avail                    \
                     ? n                                          \
                     : (interr("bad FPSRC_TYPE index", n, 4), 0)] \
      .type
#define FPSRC_SPTR(n)                                             \
  fpsrc.stg_base[n >= 0 && n < fpsrc.stg_avail                    \
                     ? n                                          \
                     : (interr("bad FPSRC_SPTR index", n, 4), 0)] \
      .sptr
#define FPSRC_PTELIST(n)                                             \
  fpsrc.stg_base[n >= 0 && n < fpsrc.stg_avail                       \
                     ? n                                             \
                     : (interr("bad FPSRC_PTELIST index", n, 4), 0)] \
      .ptelist
#else
#define FPTE_NEXT(n) fpte.stg_base[n].next
#define FPTE_TYPE(n) fpte.stg_base[n].type
#define FPTE_SPTR(n) fpte.stg_base[n].sptr
#define FPTE_VAL(n) fpte.stg_base[n].val
#define FPTE_STRIDE(n) fpte.stg_base[n].stride

#define FPSRC_NEXT(n) fpsrc.stg_base[n].next
#define FPSRC_TYPE(n) fpsrc.stg_base[n].type
#define FPSRC_SPTR(n) fpsrc.stg_base[n].sptr
#define FPSRC_PTELIST(n) fpsrc.stg_base[n].ptelist
#endif

/*
 * initialize F90 final data structures
 */
static int last_pta = 0;

static void
f90_init(void)
{
  int stdx;
  int ptr_assgns = 0;
  for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
    int astx = STD_AST(stdx);
    STD_PTA(stdx) = 0;
    if (A_TYPEG(astx) == A_ICALL && A_OPTYPEG(astx) == I_PTR2_ASSIGN)
      ptr_assgns++;
  }
  STG_ALLOC(fpte, ptr_assgns * gpsd.nslots + gpsd.nslots + 100);
  STG_ALLOC(fpsrc, ptr_assgns * gpsd.nslots + gpsd.nslots + 100);
  last_pta = 0;
} /* f90_init */

/*
 * clean up when done
 */
void
f90_fini_pointsto(void)
{
  int stdx;
  for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
    STD_PTA(stdx) = 0;
  }
  STG_DELETE(fpsrc);
  fpsrc.stg_base = NULL;
  fpsrc.stg_avail = 0;
  fpsrc.stg_size = 0;
  STG_DELETE(fpte);
  fpte.stg_base = NULL;
  fpte.stg_avail = 0;
  fpte.stg_size = 0;
} /* f90_fini_pointsto */

/*
 * add new fpsrc entry, link into a list of them for this STD
 */
static int
add_fpsrc(int psdx, int fpsrcx)
{
  int n;
  switch (PSD_TYPE(psdx)) {
  case TT_PSYM:
  case TT_ISYM:
  case TT_MEM:
  case TT_GLOB:
    n = STG_NEXT(fpsrc);
    switch (PSD_TYPE(psdx)) {
    case TT_PSYM:
    case TT_ISYM:
      FPSRC_TYPE(n) = PSD_TYPE(psdx);
      FPSRC_SPTR(n) = PSD_SYM(psdx);
      break;
    case TT_MEM:
      FPSRC_TYPE(n) = TT_ISYM;
      for (; PSD_PARENT(psdx); psdx = PSD_PARENT(psdx))
        ;
      FPSRC_SPTR(n) = PSD_SYM(psdx);
      break;
    case TT_GLOB:
      FPSRC_TYPE(n) = TT_GLOB;
      FPSRC_SPTR(n) = 0;
      break;
    }
    FPSRC_PTELIST(n) = 0;
    FPSRC_NEXT(n) = fpsrcx;
    return n;
  default:
    return fpsrcx;
  }
} /* add_fpsrc */

static void
add_fpte(int ptex, int fpsrcx)
{
  int n;
  switch (TTMASK(TPTE_TYPE(ptex))) {
  case TT_PSYM:
  case TT_ISYM:
  case TT_ANON:
  case TT_GDYN:
  case TT_IGDYN:
  case TT_LDYN:
  case TT_ILDYN:
  case TT_CON:
  case TT_NLOC:
    n = STG_NEXT(fpte);
    FPTE_TYPE(n) = TTMASK(TPTE_TYPE(ptex));
    FPTE_SPTR(n) = TPTE_SPTR(ptex);
    FPTE_VAL(n) = TPTE_VAL(ptex);
    FPTE_STRIDE(n) = TPTE_STRIDE(ptex);
    FPTE_NEXT(n) = FPSRC_PTELIST(fpsrcx);
    FPSRC_PTELIST(fpsrcx) = n;
    return;
  default:
    return;
  }
} /* add_fpte */

#if DEBUG /* { */

static FILE *dfile;

static void
putsrc(int fpsrcx)
{
  switch (FPSRC_TYPE(fpsrcx)) {
  case TT_PSYM:
    fprintf(dfile, "%d:%s", FPSRC_SPTR(fpsrcx), SYMNAME(FPSRC_SPTR(fpsrcx)));
    break;
  case TT_ISYM:
    fprintf(dfile, "%d:%s?", FPSRC_SPTR(fpsrcx), SYMNAME(FPSRC_SPTR(fpsrcx)));
    break;
  case TT_IND:
    fprintf(dfile, "?ind");
    break;
  case TT_IIND:
    fprintf(dfile, "?iind");
    break;
  case TT_MEM:
    fprintf(dfile, "%d:%s.member", FPSRC_SPTR(fpsrcx),
            SYMNAME(FPSRC_SPTR(fpsrcx)));
    break;
  case TT_GLOB:
    fprintf(dfile, ".glob.");
    break;
  case TT_NLOC:
    fprintf(dfile, ".nloc!.");
    break;
  case TT_UNINIT:
    fprintf(dfile, "Uninit!");
    break;
  case TT_UNK:
    fprintf(dfile, "Unknown!");
    break;
  default:
    fprintf(dfile, ".??%d:%d.", FPSRC_TYPE(fpsrcx), FPSRC_SPTR(fpsrcx));
    break;
  }
} /* putsrc */

static void
putfpte(int fptex)
{
  fprintf(dfile, "[%d]", fptex);
  switch (FPTE_TYPE(fptex)) {
  case TT_UNINIT:
    fprintf(dfile, "&.uninit.");
    break;
  case TT_UNK:
    fprintf(dfile, "&.unk.");
    break;
  case TT_PSYM:
    fprintf(dfile, "&%d:%s", FPTE_SPTR(fptex), SYMNAME(FPTE_SPTR(fptex)));
    break;
  case TT_ISYM:
    fprintf(dfile, "&%d:%s?", FPTE_SPTR(fptex), SYMNAME(FPTE_SPTR(fptex)));
    break;
  case TT_ANON:
    fprintf(dfile, "&anon:%d", FPTE_VAL(fptex));
    break;
  case TT_GDYN:
    fprintf(dfile, "&dyn:%d", FPTE_VAL(fptex));
    break;
  case TT_IGDYN:
    fprintf(dfile, "&dyn:%d?", FPTE_VAL(fptex));
    break;
  case TT_LDYN:
    fprintf(dfile, "&ldyn:%d", FPTE_VAL(fptex));
    break;
  case TT_ILDYN:
    fprintf(dfile, "&ldyn:%d?", FPTE_VAL(fptex));
    break;
  case TT_CON:
    fprintf(dfile, "&con:%d", FPTE_VAL(fptex));
    break;
  case TT_NLOC:
    fprintf(dfile, "&.nloc.");
    break;
  case TT_IND:
    /* should not appear */
    fprintf(dfile, "?ind");
    break;
  case TT_IIND:
    /* should not appear */
    fprintf(dfile, "?iind");
    break;
  case TT_MEM:
    /* should not appear */
    fprintf(dfile, "?mem");
    break;
  case TT_GLOB:
    /* should not appear */
    fprintf(dfile, "?glob");
    break;
  default:
    /* should not appear */
    fprintf(dfile, "???%d:%d", FPTE_TYPE(fptex), FPTE_VAL(fptex));
    break;
  }
  if (FPTE_STRIDE(fptex) > 0) {
    fprintf(gbl.dbgfil, "(::%d)", FPTE_STRIDE(fptex));
  }
} /* putfpte */

void
putstdpta(int stdx)
{
  int fpsrcx, fptex;
  dfile = gbl.dbgfil ? gbl.dbgfil : stderr;
  if (STD_PTA(stdx)) {
    fprintf(dfile, "          pta(%d @ %d) ", stdx, STD_PTA(stdx));
    for (fpsrcx = STD_PTA(stdx); fpsrcx; fpsrcx = FPSRC_NEXT(fpsrcx)) {
      fprintf(dfile, "; ");
      putsrc(fpsrcx);
      fprintf(dfile, "=>");
      for (fptex = FPSRC_PTELIST(fpsrcx); fptex; fptex = FPTE_NEXT(fptex)) {
        putfpte(fptex);
      }
    }
    fprintf(dfile, "\n");
  }
} /* putstdpta */

#endif /* } debug */

static bool
different_pta(int pta1, int pta2)
{
  int p1, p2;
  for (; pta1 && pta2; pta1 = FPSRC_NEXT(pta1), pta2 = FPSRC_NEXT(pta2)) {
    if (FPSRC_TYPE(pta1) != FPSRC_TYPE(pta2))
      return TRUE;
    if (FPSRC_SPTR(pta1) != FPSRC_SPTR(pta2))
      return TRUE;
    for (p1 = FPSRC_PTELIST(pta1), p2 = FPSRC_PTELIST(pta2); p1 && p2;
         p1 = FPTE_NEXT(p1), p2 = FPTE_NEXT(p2)) {
      if (FPTE_TYPE(p1) != FPTE_TYPE(p2))
        return TRUE;
      if (FPTE_SPTR(p1) != FPTE_SPTR(p2))
        return TRUE;
      if (FPTE_VAL(p1) != FPTE_VAL(p2))
        return TRUE;
      if (FPTE_STRIDE(p1) != FPTE_STRIDE(p2))
        return TRUE;
    }
    if (p1 || p1)
      return TRUE;
  }
  if (pta1 || pta2)
    return TRUE;
  /* identical lists */
  return FALSE;
} /* different_pta */

/*
 * Make a copy of the current list for the current statement
 * we can use the same list as for the last std if 'change' == 0
 */

static void
set_std_pta(int stdx, int change)
{
  int sl, psdx, ptex, this_pta, nthis_pta, save_fpte_avail, save_fpsrc_avail;
  if (last_pta && !change) {
    STD_PTA(stdx) = last_pta;
  } else {
    this_pta = 0;
    save_fpte_avail = fpte.stg_avail;
    save_fpsrc_avail = fpsrc.stg_avail;
    for (sl = 1; sl < gpsd.nslots; ++sl) {
      if (LHEAD(sl) != TTE_NULL && LHEAD(sl) != TPTE_UNK) {
        psdx = SLOT(sl);
        nthis_pta = add_fpsrc(psdx, this_pta);
        if (nthis_pta != this_pta) {
          this_pta = nthis_pta;
          for (ptex = LHEAD(sl); ptex > 0; ptex = TPTE_NEXT(ptex)) {
            add_fpte(ptex, this_pta);
          }
        }
      }
    }
    if (!different_pta(this_pta, last_pta)) {
      fpte.stg_avail = save_fpte_avail;
      fpsrc.stg_avail = save_fpsrc_avail;
      this_pta = last_pta;
    }
    STD_PTA(stdx) = this_pta;
    last_pta = this_pta;
  }
} /* set_std_pta */

/*
 * used externally
 *  using the previously collected pointer target information,
 *  try to determine whether the pointer at ptrsptr might point to
 *  the symbol targetsptr, or, if targetsptr is itself a pointer,
 *  might point to the same target as targetsptr
 * return TRUE if we know nothing, or we know (or believe) they might conflict
 * return FALSE if we can prove they don't conflict
 *
 * arguments are the std indices for the pointer and target references,
 * the base SPTRs of the references,
 * whether the target reference has the TARGET attribute
 * and whether the target also has the pointer attribute
 *
 * possible targets are:
 *  psym - precise symbol, with symbol sptr
 *  isym - imprecise symbol, perhaps a member of derived type, with base sptr
 *  anon - some anonymous symbol, we have no more information, but an
 *identifying tag
 *         this information comes from IPA
 *  nloc - nonlocal targets - anything but a local symbol, used for initial
 *values
 *  ldyn - locally allocated dynamic memory, with unique tag for each allocate
 *point
 *  gdyn - dynamically allocated memory from somewhere outside this routine
 *         from IPA
 *  con  - used for NULLIFY statements, DEALLOCATE statements, essentially null
 *pointer
 *  unk  - unknown
 *
 * possible conflicts for pointer and target are:
 *  ptr
 *	psym	conflict if psym sptr matches target
 *	isym	conflict if psym sptr matches target
 *	anon	no conflict - IPA only uses anon when this routine does not
 *see the symbol
 *	nloc	conflict if target is not a local (common, or host)
 *	ldyn	no conflict
 *	gdyn	no conflict
 *	con	no conflict (null doesn't conflict with anything)
 *	unk	conflict
 *	default	conflict
 *
 * possible conflicts for two pointers are:
 *
 *		psym	isym	anon	nloc	ldyn	gdyn	con
 *unk	default
 *	psym	MAY(1)	MAY(1)	no	MAY(2)	no	no
 *never	YES	YES
 *	isym	MAY(1)	MAY(1)	no	MAY(2)	no	no
 *never	YES	YES
 *	anon	no	no	MAY(3)	YES(4)	no	no	never
 *YES	YES
 *	nloc	MAY(2)	MAY(2)	YES(4)	YES	no	YES(4)
 *never	YES	YES
 *	ldyn	no	no	no	no	MAY(3)	no	never
 *YES	YES
 *	gdyn	no	no	no	YES(4)	no	MAY(3)	never
 *YES	YES
 *	con	never	never	never	never	never	never
 *never	never	never
 *	unk	YES	YES	YES	YES	YES	YES	never
 *YES	YES
 *	default	YES	YES	YES	YES	YES	YES	never
 *YES	YES
 *
 * this matrix should be symmetric
 * notes:
 * (1) two symbol targets conflict if the sptrs are the same
 * (2) symbol and nloc conflict if the symbol is not local
 * (3) two anons, ldyns, gdyns conflict if the unique value is the same
 * (4) nloc conflicts with anon and gdyn because nloc is a nonlocal target, anon
 *is
 *     an anonymous global memory location
 */

#undef TRACEFLAG
#undef TRACEBIT
#define TRACEFLAG 10
#define TRACEBIT 0x1000000

bool
pta_conflict(int ptrstdx, int ptrsptr, int targetstdx, int targetsptr,
             int targetpointer, int targettarget)
{
  int ptrsrc, targetsrc, ptrpte, targetpte;
  if (fpsrc.stg_base == NULL)
    return TRUE;
  /* see whether we have information about pointer targets for source ptrsptr at
   * statement ptrstdx */
  ptrsrc = STD_PTA(ptrstdx);
  for (; ptrsrc; ptrsrc = FPSRC_NEXT(ptrsrc)) {
    if (FPSRC_SPTR(ptrsrc) == ptrsptr)
      break;
  }
  if (ptrsrc == 0) {
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER symbol "
                          "%d:%s in std:%d\n pointer pointees are unknown\n",
              ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
              SYMNAME(targetsptr), targetstdx);
    }
#endif
    return TRUE; /* we know nothing about what this pointer targets */
  }

  if (targettarget) {
    /* 'target' has TARGET attribute. see if ptr can point to target */
    for (ptrpte = FPSRC_PTELIST(ptrsrc); ptrpte; ptrpte = FPTE_NEXT(ptrpte)) {
      switch (FPTE_TYPE(ptrpte)) {
      case TT_PSYM:
      case TT_ISYM:
        /* does the symbol match? */
        if (FPTE_SPTR(ptrpte) == targetsptr) {
#if DEBUG
          if (DBGBIT(TRACEFLAG, TRACEBIT)) {
            fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to TARGET "
                                "symbol %d:%s in std:%d\n pointer targets the "
                                "symbol directly\n",
                    ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                    SYMNAME(targetsptr), targetstdx);
          }
#endif
          return TRUE;
        }
        break;
      case TT_ANON:
      case TT_GDYN:
      case TT_IGDYN:
      case TT_LDYN:
      case TT_ILDYN:
      case TT_CON:
        /* these don't conflict with any symbol */
        break;
      case TT_NLOC:
        /* is targetsptr a nonlocal? */
        if (gbl.internal > 1 && !INTERNALG(targetsptr)) {
#if DEBUG
          if (DBGBIT(TRACEFLAG, TRACEBIT)) {
            fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to TARGET "
                                "symbol %d:%s in std:%d\n pointer targets "
                                "nonlocal memory, target is from the host\n",
                    ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                    SYMNAME(targetsptr), targetstdx);
          }
#endif
          return TRUE;
        } else {
          switch (SCG(targetsptr)) {
          case SC_CMBLK:
          case SC_EXTERN:
#if DEBUG
            if (DBGBIT(TRACEFLAG, TRACEBIT)) {
              fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to TARGET "
                                  "symbol %d:%s in std:%d\n pointer targets "
                                  "nonlocal memory, target is global\n",
                      ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                      SYMNAME(targetsptr), targetstdx);
            }
#endif
            return TRUE;
          default:;
          }
        }
        break;
      case TT_UNK:
      case TT_UNINIT:
/* might point anywhere */
#if DEBUG
        if (DBGBIT(TRACEFLAG, TRACEBIT)) {
          fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to TARGET "
                              "symbol %d:%s in std:%d pointer may point "
                              "anywhere\n",
                  ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                  SYMNAME(targetsptr), targetstdx);
        }
#endif
        return TRUE;
      case TT_IND:
      case TT_IIND:
      case TT_MEM:
      case TT_GLOB:
      default:
/* should not appear */
#if DEBUG
        if (DBGBIT(TRACEFLAG, TRACEBIT)) {
          fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to TARGET "
                              "symbol %d:%s in std:%d unexpected condition\n",
                  ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                  SYMNAME(targetsptr), targetstdx);
        }
#endif
        return TRUE;
      }
    }
  }

  if (targetpointer) {
    /* target has the pointer attribute as well */
    targetsrc = STD_PTA(targetstdx);
    for (; targetsrc; targetsrc = FPSRC_NEXT(targetsrc)) {
      if (FPSRC_SPTR(targetsrc) == targetsptr)
        break;
    }
    if (targetsrc == 0) {
#if DEBUG
      if (DBGBIT(TRACEFLAG, TRACEBIT)) {
        fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER "
                            "symbol %d:%s in std:%d\n target pointees are "
                            "unknown\n",
                ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                SYMNAME(targetsptr), targetstdx);
      }
#endif
      return TRUE; /* we know nothing about what this pointer targets */
    }

    /* compare the two pointer target lists for a match */
    for (ptrpte = FPSRC_PTELIST(ptrsrc); ptrpte; ptrpte = FPTE_NEXT(ptrpte)) {
      switch (FPTE_TYPE(ptrpte)) {
      case TT_UNK:
      case TT_UNINIT:
/* ptr might point anywhere */
#if DEBUG
        if (DBGBIT(TRACEFLAG, TRACEBIT)) {
          fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER "
                              "symbol %d:%s in std:%d\n pointer may point "
                              "anywhere\n",
                  ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                  SYMNAME(targetsptr), targetstdx);
        }
#endif
        return TRUE;
      case TT_IND:
      case TT_IIND:
      case TT_MEM:
      case TT_GLOB:
      default:
/* should not appear */
#if DEBUG
        if (DBGBIT(TRACEFLAG, TRACEBIT)) {
          fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER "
                              "symbol %d:%s in std:%d\n unexpected condition\n",
                  ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                  SYMNAME(targetsptr), targetstdx);
        }
#endif
        return TRUE;
      case TT_CON:
        if (FPTE_VAL(ptrpte) != 0) {
#if DEBUG
          if (DBGBIT(TRACEFLAG, TRACEBIT)) {
            fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER "
                                "symbol %d:%s in std:%d\n pointer may point to "
                                "nonnull constant\n",
                    ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                    SYMNAME(targetsptr), targetstdx);
          }
#endif
          return TRUE;
        }
        break;
      case TT_PSYM:
      case TT_ISYM:
      case TT_ANON:
      case TT_GDYN:
      case TT_IGDYN:
      case TT_LDYN:
      case TT_ILDYN:
      case TT_NLOC:
        break;
      }
      for (targetpte = FPSRC_PTELIST(targetsrc); targetpte;
           targetpte = FPTE_NEXT(targetpte)) {
        switch (FPTE_TYPE(targetpte)) {
        case TT_UNK:
        case TT_UNINIT:
/* target might point anywhere */
#if DEBUG
          if (DBGBIT(TRACEFLAG, TRACEBIT)) {
            fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER "
                                "symbol %d:%s in std:%d\n target may point "
                                "anywhere\n",
                    ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                    SYMNAME(targetsptr), targetstdx);
          }
#endif
          return TRUE;
        case TT_IND:
        case TT_IIND:
        case TT_MEM:
        case TT_GLOB:
        default:
/* should not appear */
#if DEBUG
          if (DBGBIT(TRACEFLAG, TRACEBIT)) {
            fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to POINTER "
                                "symbol %d:%s in std:%d\n unexpected target "
                                "condition\n",
                    ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                    SYMNAME(targetsptr), targetstdx);
          }
#endif
          return TRUE;
        case TT_CON:
          if (FPTE_VAL(targetpte) != 0) {
#if DEBUG
            if (DBGBIT(TRACEFLAG, TRACEBIT)) {
              fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                  "POINTER symbol %d:%s in std:%d\n target may "
                                  "point to nonnull constant\n",
                      ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                      SYMNAME(targetsptr), targetstdx);
            }
#endif
            return TRUE;
          }
          break;
        case TT_PSYM:
        case TT_ISYM:
        case TT_ANON:
        case TT_GDYN:
        case TT_IGDYN:
        case TT_LDYN:
        case TT_ILDYN:
        case TT_NLOC:
          break;
        }

        switch (FPTE_TYPE(ptrpte)) {
        case TT_PSYM:
        case TT_ISYM:
          switch (FPTE_TYPE(targetpte)) {
          case TT_PSYM:
          case TT_ISYM:
            /* do the symbols match? */
            if (FPTE_SPTR(ptrpte) == FPTE_SPTR(targetpte)) {
#if DEBUG
              if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                    "POINTER symbol %d:%s in std:%d\n both may "
                                    "point to %d:%s\n",
                        ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                        SYMNAME(targetsptr), targetstdx, FPTE_SPTR(ptrpte),
                        SYMNAME(FPTE_SPTR(ptrpte)));
              }
#endif
              return TRUE;
            }
            break;
          case TT_NLOC:
            /* is ptr symbol a nonlocal? */
            if (gbl.internal > 1 && !INTERNALG(FPTE_SPTR(ptrpte))) {
#if DEBUG
              if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                    "POINTER symbol %d:%s in std:%d\n pointer "
                                    "may point to %d:%s, target may point to "
                                    "nonlocal\n",
                        ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                        SYMNAME(targetsptr), targetstdx, FPTE_SPTR(ptrpte),
                        SYMNAME(FPTE_SPTR(ptrpte)));
              }
#endif
              return TRUE;
            } else {
              switch (SCG(FPTE_SPTR(ptrpte))) {
              case SC_CMBLK:
              case SC_EXTERN:
#if DEBUG
                if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                  fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                      "POINTER symbol %d:%s in std:%d\n "
                                      "pointer may point to %d:%s, target may "
                                      "point to common\n",
                          ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                          SYMNAME(targetsptr), targetstdx, FPTE_SPTR(ptrpte),
                          SYMNAME(FPTE_SPTR(ptrpte)));
                }
#endif
                return TRUE;
              default:;
              }
            }
            break;
            /* symbols do not conflict with ldyn, gdyn, anon;
             *  unk and default cases already handled */
          }
          break;
        case TT_LDYN:
        case TT_ILDYN:
          /* only conflicts with the another local dynamic and value */
          if (FPTE_TYPE(targetpte) == FPTE_TYPE(ptrpte)) {
            if (FPTE_VAL(targetpte) == FPTE_VAL(ptrpte)) {
#if DEBUG
              if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                    "POINTER symbol %d:%s in std:%d\n both "
                                    "point to local dynamically allocated "
                                    "memory\n",
                        ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                        SYMNAME(targetsptr), targetstdx);
              }
#endif
              return TRUE;
            }
          }
          /* else no conflict from this target */
          break;
        case TT_ANON:
        case TT_GDYN:
        case TT_IGDYN:
          /* conflicts with same type and value,
           * also conflicts with NLOC */
          if (FPTE_TYPE(targetpte) == FPTE_TYPE(ptrpte)) {
            if (FPTE_VAL(targetpte) == FPTE_VAL(ptrpte)) {
#if DEBUG
              if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                    "POINTER symbol %d:%s in std:%d\n both may "
                                    "point to globally allocated memory\n",
                        ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                        SYMNAME(targetsptr), targetstdx);
              }
#endif
              return TRUE;
            }
          } else if (FPTE_TYPE(targetpte) == TT_NLOC) {
#if DEBUG
            if (DBGBIT(TRACEFLAG, TRACEBIT)) {
              fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                  "POINTER symbol %d:%s in std:%d\n pointer "
                                  "may point to globally allocated memory, "
                                  "target may point to nonlocal\n",
                      ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                      SYMNAME(targetsptr), targetstdx);
            }
#endif
            return TRUE;
          }
          break;
        case TT_CON:
          /* NULL pointer doesn't conflict */
          break;
        case TT_NLOC:
          /* conflicts with nonlocal, anon, gdyn, any nonlocal symbol */
          switch (FPTE_TYPE(targetpte)) {
          case TT_NLOC:
          case TT_GDYN:
          case TT_IGDYN:
          case TT_ANON:
#if DEBUG
            if (DBGBIT(TRACEFLAG, TRACEBIT)) {
              fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                  "POINTER symbol %d:%s in std:%d\n pointer "
                                  "may point to nonlocal, target may point to "
                                  "nonlocal or global dynamic or anonymous\n",
                      ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                      SYMNAME(targetsptr), targetstdx);
            }
#endif
            return TRUE;
          case TT_PSYM:
          case TT_ISYM:
            /* is targetsptr a nonlocal? */
            if (gbl.internal > 1 && !INTERNALG(FPTE_SPTR(targetpte))) {
#if DEBUG
              if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                    "POINTER symbol %d:%s in std:%d\n pointer "
                                    "may point to nonlocal, target may point "
                                    "to %d:%s which is from the host\n",
                        ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                        SYMNAME(targetsptr), targetstdx, FPTE_SPTR(targetpte),
                        SYMNAME(FPTE_SPTR(targetpte)));
              }
#endif
              return TRUE;
            } else {
              switch (SCG(FPTE_SPTR(targetpte))) {
              case SC_CMBLK:
              case SC_EXTERN:
#if DEBUG
                if (DBGBIT(TRACEFLAG, TRACEBIT)) {
                  fprintf(gbl.dbgfil, "pointer %d:%s in std:%d may point to "
                                      "POINTER symbol %d:%s in std:%d\n "
                                      "pointer may point to nonlocal, target "
                                      "may point to %d:%s which is external\n",
                          ptrsptr, SYMNAME(ptrsptr), ptrstdx, targetsptr,
                          SYMNAME(targetsptr), targetstdx, FPTE_SPTR(targetpte),
                          SYMNAME(FPTE_SPTR(targetpte)));
                }
#endif
                return TRUE;
              default:;
              }
            }
            break;
          }
          break;
        }
      }
    }
  }
  return FALSE;
} /* pta_conflict */

/*
 * used externally
 *  using the previously collected pointer target information,
 *  try to determine whether the pointer at ptrsptr always points to aligned
 *  data.
 * return TRUE if we know the pointer target is always quad aligned
 * return FALSE if we don't know or know otherwise
 *
 * arguments are the std indices for the pointer reference,
 * and the base SPTRs of the reference
 *
 * possible targets are:
 *  psym - precise symbol, with symbol sptr
 *         the target is aligned if the symbol is aligned and stride is one
 *  isym - imprecise symbol, perhaps a member of derived type, with base sptr
 *         unaligned
 *  anon - some anonymous symbol, we have no more information, but an
 * identifying tag
 *         this information comes from IPA
 *         unaligned, unless we enhance this information from IPA
 *  nloc - nonlocal targets - anything but a local symbol, used for initial
 * values
 *         unaligned
 *  ldyn - locally allocated dynamic memory, with unique tag for each allocate
 * point
 *         aligned or unaligned, depending on whether it is precise and stride
 * is one
 *  gdyn - dynamically allocated memory from somewhere outside this routine
 *         from IPA
 *         usually unaligned, unless we know it's precise and stride is one
 *  con  - used for NULLIFY statements, DEALLOCATE statements, essentially null
 * pointer
 *         shouldn't happen, treat as aligned
 *  unk  - unknown
 *         unaligned
 */
bool
pta_aligned(int ptrstdx, int ptrsptr)
{
  int ptrsrc, ptrpte, sptr;
  if (fpsrc.stg_base == NULL)
    return FALSE;
  /* see whether we have information about pointer targets for source ptrsptr at
   * statement ptrstdx */
  ptrsrc = STD_PTA(ptrstdx);
  for (; ptrsrc; ptrsrc = FPSRC_NEXT(ptrsrc)) {
    if (FPSRC_SPTR(ptrsrc) == ptrsptr)
      break;
  }
  if (ptrsrc == 0)
    return FALSE; /* we know nothing about what this pointer targets */

  for (ptrpte = FPSRC_PTELIST(ptrsrc); ptrpte; ptrpte = FPTE_NEXT(ptrpte)) {
    switch (FPTE_TYPE(ptrpte)) {
    case TT_PSYM:
      /* might be aligned if the symbol is */
      sptr = FPTE_SPTR(ptrpte);
      if (!QALNG(sptr))
        return FALSE;
      if (FPTE_STRIDE(ptrpte) != 1)
        return FALSE;
      break;
    case TT_GDYN:
      if (FPTE_STRIDE(ptrpte) != 1)
        return FALSE;
      break;
    case TT_LDYN:
      if (FPTE_STRIDE(ptrpte) != 1)
        return FALSE;
      break;
    case TT_CON:
      break;
    case TT_ISYM:  /* somewhere in the symbol */
    case TT_ANON:  /* don't know where */
    case TT_ILDYN: /* somewhere in the allocated space */
    case TT_IGDYN:
      return FALSE;
    case TT_NLOC:   /* some nonlocal symbol */
    case TT_UNK:    /* anywhere */
    case TT_UNINIT: /* anywhere */
      /* might point anywhere */
      return FALSE;
    case TT_IND: /* some indirect location, shouldn't happen here */
    case TT_IIND:
    case TT_MEM:
    case TT_GLOB:
    default:
      /* should not appear */
      return FALSE;
    }
  }
  /* no show stoppers found == no misaligned targets found */
  return TRUE;
} /* pta_aligned */

/*
 * used externally
 *  using previous collected pointer target information,
 *  determine whether the pointer target is always stride-1, meaning
 *  for instance that it can be passed directly to an assumed-shape argument
 *  without copying to a temp
 */
bool
pta_stride1(int ptrstdx, int ptrsptr)
{
  int ptrsrc, ptrpte;
  if (fpsrc.stg_base == NULL)
    return FALSE;
  /* see whether we have information about pointer targets for source ptrsptr at
   * statement ptrstdx */
  ptrsrc = STD_PTA(ptrstdx);
  for (; ptrsrc; ptrsrc = FPSRC_NEXT(ptrsrc)) {
    if (FPSRC_SPTR(ptrsrc) == ptrsptr)
      break;
  }
  if (ptrsrc == 0)
    return FALSE; /* we know nothing about what this pointer targets */

  for (ptrpte = FPSRC_PTELIST(ptrsrc); ptrpte; ptrpte = FPTE_NEXT(ptrpte)) {
    switch (FPTE_TYPE(ptrpte)) {
    case TT_PSYM:
    case TT_ISYM: /* somewhere in the symbol */
      if (FPTE_STRIDE(ptrpte) != 1)
        return FALSE;
      break;
    case TT_GDYN:
    case TT_IGDYN:
      if (FPTE_STRIDE(ptrpte) != 1)
        return FALSE;
      break;
    case TT_LDYN:
    case TT_ILDYN: /* somewhere in the allocated space */
      if (FPTE_STRIDE(ptrpte) != 1)
        return FALSE;
      break;
    case TT_CON:
      break;
    case TT_ANON:   /* don't know where */
    case TT_NLOC:   /* some nonlocal symbol */
    case TT_UNK:    /* anywhere */
    case TT_UNINIT: /* anywhere */
      /* might point anywhere */
      return FALSE;
    case TT_IND: /* some indirect location, shouldn't happen here */
    case TT_IIND:
    case TT_MEM:
    case TT_GLOB:
    default:
      /* should not appear */
      return FALSE;
    }
  }
  /* no show stoppers found == only stride-1 targets found */
  return TRUE;
} /* pta_stride1 */

/*
 * used externally
 *  using previous collected pointer target information,
 *  determine whether there is pointer target information for this
 * statement/sptr
 *  return a tag and id to describe it.
 *  tag may be one of:
 *   0 = no more information
 *   1 = unknown target
 *   2 = local dynamically allocated memory (tag disambiguates)
 *   3 = global dynamically allocated memory (tag disambiguates)
 *   4 = nonlocal memory (tag disambiguates)
 *   5 = precise symbol target (tag is sptr)
 *   6 = imprecise symbol target (tag is sptr)
 */
int
pta_target(int ptrstdx, int ptrsptr, int *ptag, int *pid)
{
  static int prev_func_count = 0, prevptrstdx = 0, prevptrsptr = 0, ptrsrc = 0,
             ptrpte = 0;
  if (fpsrc.stg_base == NULL)
    return 0;
  if (prevptrstdx == ptrstdx && prevptrsptr == ptrsptr &&
      prev_func_count == gbl.func_count) {
    ptrpte = FPTE_NEXT(ptrpte);
  } else {
    prev_func_count = gbl.func_count;
    ptrpte = 0;
    prevptrstdx = 0;
    prevptrsptr = 0;
    /* see whether we have information about pointer targets for source ptrsptr
     * at
     * statement ptrstdx */
    ptrsrc = STD_PTA(ptrstdx);
    for (; ptrsrc; ptrsrc = FPSRC_NEXT(ptrsrc)) {
      if (FPSRC_SPTR(ptrsrc) == ptrsptr)
        break;
    }
    if (ptrsrc == 0) {
      return 0; /* we know nothing about what this pointer targets */
    }
    prevptrstdx = ptrstdx;
    prevptrsptr = ptrsptr;
    ptrpte = FPSRC_PTELIST(ptrsrc);
  }

  if (ptrpte == 0) {
    prevptrstdx = 0;
    prevptrsptr = 0;
    return 0; /* we know nothing about what this pointer targets */
  }

  switch (FPTE_TYPE(ptrpte)) {
  case TT_PSYM:
    *ptag = 5;
    *pid = FPTE_SPTR(ptrpte);
    return 1;
  case TT_ISYM: /* somewhere in the symbol */
    *ptag = 6;
    *pid = FPTE_SPTR(ptrpte);
    return 1;
  case TT_GDYN:
  case TT_IGDYN:
    *ptag = 3;
    *pid = FPTE_VAL(ptrpte);
    return 1;
  case TT_LDYN:
  case TT_ILDYN: /* somewhere in the allocated space */
    *ptag = 2;
    *pid = FPTE_VAL(ptrpte);
    return 1;
  case TT_CON:
    *ptag = 1;
    *pid = 0;
    return 1;
  case TT_ANON:   /* don't know where */
  case TT_NLOC:   /* some nonlocal symbol */
  case TT_UNK:    /* anywhere */
  case TT_UNINIT: /* anywhere */
    *ptag = 1;
    *pid = 0;
    return 1;
  case TT_IND: /* some indirect location, shouldn't happen here */
  case TT_IIND:
  case TT_MEM:
  case TT_GLOB:
  default:
    *ptag = 1;
    *pid = 0;
    return 1;
  }
} /* pta_target */

#undef TRACEFLAG
#undef TRACEBIT
#define TRACEFLAG 10
#define TRACEBIT 0x100000

/*
 * allocate storage
 *  pointer source descriptor, with hash table
 *  temporary pointer descriptors
 *  pointer assignments
 *  list of pointer assignments for stmts
 */
static void
init_points_to_anal(void)
{
  int a;
  STG_ALLOC(gpte, 100);
  gpte.xstg_free = TTE_NULL;
  gpte.stg_avail = 2;
  /* entry zero is TPTE_UNK */
  STG_CLEAR_ALL(gpte);

  STG_ALLOC(apte, 100);
  BZERO(aptehsh, int, APTEHSZ);

  a = get_apte(TT_UNK, 0, 0, 0);
  TPTE_APTEX(TPTE_UNK) = a;
  TPTE_NEXT(TPTE_UNK) = TPTE_NULL;

  a = get_apte(TT_UNINIT, 0, 0, 0);
  TPTE_APTEX(TPTE_UNINIT) = a;
  TPTE_NEXT(TPTE_UNINIT) = TPTE_NULL;

  STG_ALLOC(gpsd, nmeb.stg_avail < 2 ? 2 : nmeb.stg_avail);
  BZERO(gpsd.stg_base, PSD, 2);
  BZERO(psdhsh, int, PSDHSZ);
  gpsd.stg_avail = 2;
  PSD_TYPE(1) = TT_UNINIT;
  PSD_TYPE(0) = TT_UNK;
  gpsd.nslots = 1;
  gpsd.slot = NULL;

  STG_ALLOC(gtpd, 20);
  gtpd.stg_avail = 3;
  STG_CLEAR_ALL(gtpd);
  TTYPE(2) = TT_GLOB;
  TTYPE(1) = TT_UNINIT;
  TLINK(1) = 1;
  TTYPE(0) = TT_UNK;

  STG_ALLOC(as, 100);

  STG_ALLOC(ganon, 20);
  BZERO(ganon.stg_base, int, 2);

  STG_ALLOC(gdyn, 20);
  BZERO(gdyn.stg_base, int, 2);
  gdyn.local_dyn = 1;

  FIRSTAS(0) = 0;
} /* init_points_to_anal */

/*
 * allocate storage
 *  flow graph
 *  reverse depth-first ordering of flow graph nodes
 */
static void
init_points_to_prop(void)
{
  int savex6;

  savex6 = flg.x[6]; /* disable flow graph changes here */
  flg.x[6] |= 0x80000000;
  optshrd_init();
  flowgraph();
  flg.x[6] = savex6;
  NEW(rdfo, int, opt.num_nodes + 1);
  NEW(rdfonum, int, opt.num_nodes + 1);
  BZERO(rdfonum, int, opt.num_nodes + 1);
  rdfocount = 0;
  buildrdfo(VTX_NODE(1));
  gpte.save_stg_avail = gpte.stg_avail;
  gpte.save_stg_free = gpte.xstg_free;
} /* init_points_to_prop */

/*
 * free storage
 */
static void
fini_points_to_anal(void)
{
  STG_DELETE(gtpd);
} /* fini_points_to_anal */

/*
 * free storage
 */
static void
fini_points_to_prop(void)
{
  if (nodeoffset) {
    FREE(nodeoffset);
    nodeoffset = NULL;
  }
  STG_DELETE(head);
  if (localhead) {
    FREE(localhead);
    localhead = NULL;
  }
  if (gpsd.slot) {
    FREE(gpsd.slot);
    gpsd.slot = NULL;
  }
  FREE(rdfonum);
  rdfonum = NULL;
  FREE(rdfo);
  rdfo = NULL;

  optshrd_end();
  gpte.stg_avail = gpte.save_stg_avail;
  gpte.xstg_free = gpte.save_stg_free;
} /* fini_points_to_prop */

/*
 * free storage
 */
void
fini_points_to_all(void)
{
  STG_DELETE(gdyn);
  STG_DELETE(ganon);
  STG_DELETE(as);
  STG_DELETE(gtpd);

  STG_DELETE(gpsd);
  STG_DELETE(apte);
  STG_DELETE(gpte);
} /* fini_points_to_all */

/*
 * collect info and create the pointer pseudo assignments for later propagation
 */
void
points_to_anal(void)
{
  int stdx, a;
#if DEBUG
  if (gbl.dbgfil == NULL)
    gbl.dbgfil = stderr;
#endif
  init_points_to_anal();

  /* create a global pointer source descriptor */
  /* this is the pseudo-variable to collect any anonymous global changes */
  add_psd(0, TT_GLOB, 0, 0, 1);
  /* find the pointer assignments for each of the statements */
  for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
    find_pointer_assignments_f90(stdx);
  }
/* ### here, we should be creating initialization pseudo-assignments
 * to correspond to the initial values of globals or function arguments */
  /* clean up */
  fini_points_to_anal();
  if (as.stg_avail == 1) {
    /* nothing to learn */
    Trace(("pointer target analysis has no pointer assignments\n"));
    return;
  }
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT)) {
    if (FIRSTAS(0)) {
      fprintf(gbl.dbgfil, "\nstmt:%d pointer assignments\n", 0);
      for (a = FIRSTAS(0); a > 0; a = ASNEXT(a)) {
        putassign(a);
      }
    }
    for (stdx = STD_NEXT(0); stdx; stdx = STD_NEXT(stdx)) {
      if (FIRSTAS(stdx)) {
        fprintf(gbl.dbgfil, "\nnode:%d pointer assignments\n", stdx);
        for (a = FIRSTAS(stdx); a > 0; a = ASNEXT(a)) {
          putassign(a);
        }
      }
    }
    fprintf(gbl.dbgfil, "\n");
    fprintf(gbl.dbgfil,
            "POINTSTO - %5d NME, %5d PSD, %5d SLOTS, %5d APTE, %5d BLOCKS\n",
            nmeb.stg_avail, gpsd.stg_avail, gpsd.nslots, apte.stg_avail,
            opt.num_nodes);
  }
#endif
} /* points_to_anal */

/** \brief Do dataflow analysis to determine pointer target information
    store pointer target information at loads and stores

    - build flow graph
    - build reverse depth-first ordering
    - build pseudo-assignments to represent pointer modifications
    - use data-flow analysis to propagate pointer information around flow graph
    - modify the ILI/NME to represent the propagated information

    For F90.
 */
void
points_to(void)
{
  int r, v, rdfohigh, nextrdfohigh, rdfolow, nextrdfolow;
  PSI_P succ;
  int sl, psdx, offset, iteration, asx;
  int stdx, change;

  /* initialize */
  init_points_to_prop();

  /* can we afford to build the data flow equations?
   * we will need a word for each critical basic block for a
   * pointer target list head for each PSD; also a structure to
   * to store lists of actual pointer targets */
  head.stg_size = opt.num_nodes * gpsd.nslots + 1;
  if (head.stg_size > 1000000) {
    /* abort */
    Trace(("pointer target analysis is too expensive, abort\n"));
    fini_points_to_prop();
    return;
  }
  STG_ALLOC(head, head.stg_size);
  NEW(nodeoffset, int, opt.num_nodes + 1);
  BZERO(nodeoffset, int, opt.num_nodes + 1);
  NEW(localhead, int, gpsd.nslots + 1);
  NEW(gpsd.slot, int, gpsd.nslots + 1);
  BZERO(gpsd.slot, int, gpsd.nslots + 1);

  /* assign a slot number to each pointer, called a PSD or pointer source
   * descriptor.
   * For Fortran, these are pretty simple */
  for (psdx = 1; psdx < gpsd.stg_avail; ++psdx) {
    if (PSD_SLOT(psdx))
      SLOT(PSD_SLOT(psdx)) = psdx;
  }

  offset = 1;
  /* to simplify memory allocation, we allocate one big vector to hold
   * a matrix of information.  Here, we give the index into the vector
   * for the information for each flow graph node */
  /* fill in nodeoffset */
  for (r = rdfocount; r > 0; --r) {
    v = RDFO(r);
    nodeoffset[v] = offset;
    offset += gpsd.nslots;
    /* default information is 'TOP', points to nothing */
    for (sl = 1; sl < gpsd.nslots; ++sl)
      HEAD(v, sl) = TTE_NULL;
  }
  v = RDFO(rdfocount); /* entry node */
                       /*
                        * interpret the init assignments at the start node
                        * Start out by assuming the pointers are all empty
                        */
  for (sl = 0; sl < gpsd.nslots; ++sl) {
    HEAD(v, sl) = TPTE_NULL;
    LHEAD(sl) = TPTE_NULL;
  }
  for (asx = FIRSTAS(0); asx > 0 && ASTYPE(asx) == AS_INIT; asx = ASNEXT(asx)) {
    interpret(asx);
  }
  /*
   * For any nonlocal pointers that have no initial information,
   * assume they start out pointing to any nonlocal
   */
  for (sl = 1; sl < gpsd.nslots; ++sl) {
    psdx = SLOT(sl);
    if (LHEAD(sl) != TPTE_NULL) {
      HEAD(v, sl) = LHEAD(sl);
    } else if (is_source_nonlocal(psdx)) {
      HEAD(v, sl) = get_pte(TT_NLOC, 0, 0, 0);
    } else {
      HEAD(v, sl) = TPTE_UNINIT;
    }
    /* reinitialize LHEAD */
    LHEAD(sl) = TPTE_NULL;
  }

  /* repeat until no changes, first iteration must visit all nodes */
  nextrdfohigh = rdfocount;
  nextrdfolow = 1;
  iteration = 0;
  do {
    ++iteration;
    rdfohigh = nextrdfohigh;
    rdfolow = nextrdfolow;
    nextrdfohigh = 0;
    nextrdfolow = rdfocount;
    Trace(("------------------------"));
    Trace(("iteration %d, node positions %d:%d", iteration, rdfohigh, rdfolow));
#if DEBUG
    check_pte("TPTE error: before iterating");
#endif
    /* visit each basic block
     * symbolically execute each assignment, see how it
     * affects the local pointer target information */
    for (r = rdfohigh; r >= rdfolow; --r) {
      v = RDFO(r);
      Trace(("%4d=r, fg:%d, bih:%d", r, v, FG_TO_BIH(v)));
      for (sl = 1; sl < gpsd.nslots; ++sl) {
        free_list_slot(sl);
        LHEAD(sl) = copy_list(HEAD(v, sl));
      }
#if DEBUG
      if (DBGBIT(TRACEFLAG, TRACEBIT)) {
        Trace(("before node %d block %d std %d:%d", v, FG_TO_BIH(v),
               FG_STDFIRST(v), FG_STDLAST(v)));
        puttargets();
      }
#endif
      for (stdx = FG_STDFIRST(v); stdx; stdx = STD_NEXT(stdx)) {
#if DEBUG
        if (DBGBIT(TRACEFLAG, TRACEBIT)) {
          dstdp(stdx);
        }
#endif
        for (asx = FIRSTAS(stdx); asx > 0; asx = ASNEXT(asx)) {
          interpret(asx);
        }
        if (stdx == FG_STDLAST(v))
          break;
      }
#if DEBUG
      if (DBGBIT(TRACEFLAG, TRACEBIT)) {
        Trace(("after node %d block %d", v, FG_TO_BIH(v)));
        puttargets();
      }
#endif
      for (succ = FG_SUCC(v); succ != PSI_P_NULL; succ = PSI_NEXT(succ)) {
        int s, sr;
        s = PSI_NODE(succ);
        Trace(("flow edge from node %d -> %d", v, s));
        if (1 /*critical(s)*/) {
          if (pte_changed(s)) {
            sr = RDFONUM(s);
            if (sr >= r && sr > nextrdfohigh) {
              /* must iterate again at least from sr */
              nextrdfohigh = sr;
              if (sr < nextrdfolow)
                nextrdfolow = sr;
            } else if (sr < rdfolow) {
              rdfolow = sr; /* keep going to 'sr' in this iteration */
            } else if (sr < nextrdfolow) {
              nextrdfolow = sr;
            }
          }
        }
      }
    }
    for (sl = 1; sl < gpsd.nslots; ++sl) {
      free_list_slot(sl);
      LHEAD(sl) = TTE_NULL;
    }
  } while (nextrdfohigh >= nextrdfolow);
#if DEBUG
  check_pte("TPTE error: done iterating");
  Trace(("POINTSTO - %5d NME, %5d PSD, %5d SLOTS, %5d APTE, %5d BLOCKS",
         nmeb.stg_avail, gpsd.stg_avail, gpsd.nslots, apte.stg_avail,
         opt.num_nodes));
#endif

  /* now modify the STDs */
  asx = 0;
  gcount.tot = 0;
  gcount.ntot = 0;
  f90_init();
  for (r = rdfocount; r >= 1; --r) {
    v = RDFO(r);
    Trace(("------------------------"));
    Trace(("insert %4d=r, fg:%d", r, v));
    for (sl = 1; sl < gpsd.nslots; ++sl) {
      free_list_slot(sl);
      LHEAD(sl) = copy_list(HEAD(v, sl));
    }
    if (asx > 0) {
      /* real error, didn't finish assignments from previous block */
      interr("pointsto: didn't finish all symbolic assignments", asx, ERR_Fatal);
    }
#if DEBUG
    if (DBGBIT(TRACEFLAG, TRACEBIT)) {
      Trace(("before node %d stds %d:%d", v, FG_STDFIRST(v), FG_STDLAST(v)));
      puttargets();
    }
#endif
    change = 1;
    for (stdx = FG_STDFIRST(v); stdx; stdx = STD_NEXT(stdx)) {
      set_std_pta(stdx, change);
#if DEBUG
      if (DBGBIT(TRACEFLAG, TRACEBIT)) {
        if (change) {
          putstdpta(stdx);
        } else {
          if (STD_PTA(stdx)) {
            fprintf(gbl.dbgfil, "pta(std=%d) same as std %d\n", stdx,
                    STD_PREV(stdx));
          }
        }
      }
#endif
      if (stdx == FG_STDLAST(v)) {
        asx = 0;
        break;
      }
      change = 0;
      for (asx = FIRSTAS(stdx); asx > 0; asx = ASNEXT(asx)) {
        if (ASTYPE(asx) != AS_INIT) {
          change = 1;
          interpret(asx);
        }
      }
    }
  }
  fini_points_to_prop();
#if DEBUG
  if (DBGBIT(TRACEFLAG, TRACEBIT)) {
    dstdps(0, 0);
  }
#endif
} /* points_to for F90 */

