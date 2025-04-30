/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef DIRECT_H_
#define DIRECT_H_

/**
   \file
   \brief directive/pragma data structures.
 */

#include "gbldefs.h"
#include "error.h"
#include "global.h"
#include "symtab.h"
#include <stdio.h>

#ifdef FE90
/* INDEPENDENT information */
typedef struct _newvar_ {
  int var;
  struct _newvar_ *next;
} NEWVAR;

typedef struct _reducvar_ {
  int var;    /* REDUCTION variable */
  int newvar; /* INDEP loop's copy of REDUCTION variable */
  struct _reducvar_ *next;
} REDUCVAR;

typedef struct _reduc_ja_spec {/* HPF/JA REDUCTION clause item */
  int var;                     /* reduction variable */
  int newvar;                  /* INDEP loop's copy of reduction variable */
  int nlocvars;                /* number of location variables */
  REDUCVAR *locvar_list;       /* [ / location variable list / ] */
  struct _reduc_ja_spec *next;
} REDUC_JA_SPEC;

typedef struct _reduc_ja { /* HPF/JA REDUCTION clause (with reduc-op) */
  int opr;                 /* if != 0, OP_xxx value */
  int intrin;              /* if != 0, sptr to intrinsic */
  REDUC_JA_SPEC *speclist; /* list of reduction variables */
  struct _reduc_ja *next;
} REDUC_JA;

typedef struct {
  NEWVAR *new_list;
  int onhome;
  REDUCVAR *reduction_list;
  REDUC_JA *reduction_ja_list;
  NEWVAR *index_list;
} INDEP_INFO;

typedef struct _hpf2_on_info_ {
  NEWVAR *new_list;
  int onhome;
  int block; /* if == 0, simple on dir */
  struct _hpf2_on_info_ *next;
} HPF2_ON_INFO;

typedef struct _index_reuse_ {/* contents of one INDEX_REUSE directive */
  int condvar;                /* if != 0, sptr to scalar logical temporary
                                 storing the 'index reuse condition' */
  NEWVAR *reuse_list;         /* list of array variables in INDEX_REUSE */
  struct _index_reuse_ *next;
} INDEX_REUSE;

#endif // FE90

typedef struct {
  /* NOTES:
   * 1.  all members must be int
   * 2.  any additions/modifications imply load_dirset() and store_dirset()
   *     in direct.c, and set_flg() in pragma.c, must be modified.
   * 3.  set_flg() cares about the order in which the members occur.
   * 4.  the member x must be the last member in this structure.
   *     DIRSET_XFLAG is x's offset (in units of ints) from the beginning
   *     of the structure.
   */
  int opt;
  int vect;
  int depchk;
  int fcon;   /* C-only, but always declared */
  int single; /* C-only, but always declared */
  int tpvalue[TPNVERSION]; /* target processor(s), for unified binary */
  int x[sizeof(flg.x) / sizeof(int)]; /* same as flg.x[...] */
} DIRSET;

#define DIRSET_XFLAG 15

typedef struct lpprg_ {/* pragma information for loops */
  int beg_line;        /* beginning line # of loop */
  int end_line;        /* ending line # of loop */
  DIRSET dirset;       /* dirset for the loop */
#ifdef FE90
  INDEP_INFO *indep;             /* locates a loop's INDEP_INFO */
  INDEX_REUSE *index_reuse_list; /* locates a loop's INDEX_REUSE */
#endif
} LPPRG;

typedef struct {/* loop pragma stack */
  int dirx;     /* index into lpg of the loop's dirset */
} LPG_STK;

/** \brief Directive structure
 */
typedef struct {
  DIRSET gbl;        /**< Holding area for global-scoped pragmas */
  DIRSET rou;        /**< Holding area for routine-scoped pragmas */
  DIRSET loop;       /**< Holding area for loop-scoped pragmas */
  DIRSET rou_begin;  /**< Pragmas which apply to the beginning of a routine.
                       *  For C, this structure must be saved away for each
                       *  function appearing in the source file.
                       */
  bool loop_flag; /**< Seen pragma with loop scope */
  bool in_loop;   /**< Currently in loop with pragmas */
  bool carry_fwd; /**< If global/routine pragma seen, must be carried
                      * forward to all outer loops which follow in the
                      * routine.
                      */
  /**
   * for C, need to allocate a DIRSET for each function -- is located
   * by the function's ENTRY aux structure and is assigned by dir_rou_end().
   *
   * for C & Fortran, need to allocate a DIRSET for a loop which has
   * pragmas associated with it.
   */
  DIRSET *stgb;
  int size;
  int avail;
  /**
   * for C & Fortran, each function is associated with a set of
   * loop pragma information. The set is organized as a table
   * and will be ordered according to occurrence of loops (with
   * associated pragmas) in the function.
   */
  struct {
    LPPRG *stgb;
    int size;
    int avail;
  } lpg;
  struct {
    LPG_STK *stgb;
    int size;
    int top;
  } lpg_stk;
#ifdef FE90
  struct {
    LPPRG *stgb;
    int size;
    int avail;
  } dynlpg;
  INDEP_INFO *indep; /**< Locates where to record an INDEPENDENT's INDEP_INFO
                      * while processing the INDEPENDENT statement;
                      * locates a loop's INDEP_INFO structure after
                      * a call to open_pragma()
                      */
  INDEX_REUSE *index_reuse_list; /**< Likewise for an INDEX_REUSE structure */
#endif
} DIRECT;

extern DIRECT direct;

#ifdef FE90
void open_dynpragma(int, int); /* ilidir.h */
void save_dynpragma(int);
void direct_loop_save(void);
#endif

void direct_export(FILE *ff);

/**
   \brief ...
 */
int direct_import(FILE *ff);

/**
   \brief ...
 */
void direct_fini(void);

/**
   \brief ...
 */
void direct_init(void);

/**
   \brief ...
 */
void direct_loop_end(int beg_line, int end_line);

/**
   \brief ...
 */
void direct_loop_enter(void);

/**
   \brief ...
 */
void direct_rou_end(void);

/**
   \brief ...
 */
void direct_rou_load(int func);

/**
   \brief ...
 */
void direct_rou_setopt(int func, int opt);

/**
   \brief ...
 */
void direct_xf(char *fn, int x, int v);

/**
   \brief ...
 */
void direct_yf(char *fn, int x, int v);

/**
   \brief ...
 */
void dirset_options(bool restore);

/**
   \brief ...
 */
void load_dirset(DIRSET *currdir);

/**
   \brief ...
 */
void store_dirset(DIRSET *currdir);

#endif // DIRECT_H_
