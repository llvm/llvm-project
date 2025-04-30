/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ILT_H_
#define ILT_H_

/** \file
 * \brief ILT data structures and definitions
 */

typedef struct {
  int ilip;
  union {
    UINT all;
    struct {
      unsigned ex : 1;
      unsigned st : 1;
      unsigned br : 1; /* ILT can branch */
      unsigned lb : 1;
      unsigned dbgline : 1;
      unsigned delete_ : 1;
      unsigned ignore : 1; /* used by hl vectorizer */
      unsigned split : 1;  /* split the loop here */
      unsigned cplx : 1;   /* pseudo store of complex store sequence */
      unsigned mcache : 1; /* store guaranteed to miss cache */
      unsigned nodel : 1;  /* fp store cannot be deleted */
      unsigned keep : 1;   /* don't delete */
      unsigned delebb : 1; /* delete store if its subsequent use appears
                            * in the same extended basic block */
      unsigned predc : 1;  /* ILT serves to combine two guards */
      unsigned eqasrt : 1; /* if set, ilt is for a store used to assert
                            * that a variable has the given value
                            */
      unsigned free : 1;   /* ilt is free - it's in the free list */
      unsigned class_ : 5; /* used in accelerator compiler */
      unsigned extra : 1;  /* also used in accelerator compiler */
      unsigned extra2 : 1; /* also used in accelerator compiler */
      unsigned inv : 1;    /* hoisted invariant */
      unsigned acc_disabled : 1; /* ILT will be disabled at GPU CG */
      unsigned spare : 7;
    } bits;
  } flags;
  int prev;
  int next;
  int order; /* used to keep track of order within a block */
  int lineno;
  int findex;
  int oldilt;
} ILT;

typedef struct {
  STG_MEMBERS(ILT);
  int curilt;
  int callfg;
  char ldvol;   /* Volatile load flag */
  char stvol;   /* Volatile store flag */
  bool qjsrfg;  /* QJSR flag */
  char privtmp; /* private temp state; DEBUG-only dmpilt() sets */
} ILTB;

#define ILT_ILIP(i) iltb.stg_base[i].ilip
#define ILT_LINENO(i) iltb.stg_base[i].lineno
#define ILT_OLDILT(i) iltb.stg_base[i].oldilt
#define ILT_FINDEX(i) iltb.stg_base[i].findex
#define ILT_PREV(i) iltb.stg_base[i].prev
#define ILT_NEXT(i) iltb.stg_base[i].next
#define ILT_ORDER(i) iltb.stg_base[i].order
#define ILT_FLAGS(i) iltb.stg_base[i].flags.all
#define ILT_EX(i) iltb.stg_base[i].flags.bits.ex
#define ILT_ST(i) iltb.stg_base[i].flags.bits.st
#define ILT_BR(i) iltb.stg_base[i].flags.bits.br
#define ILT_ACC_DISABLED(i) iltb.stg_base[i].flags.bits.acc_disabled
#define ILT_CAN_THROW(i) (0)
#define ILT_SET_CAN_THROW(i, value) ((void)0)
#define ILT_BR_OR_CAN_THROW(i) (ILT_BR(i) || ILT_CAN_THROW(i))
#define ILT_LB(i) iltb.stg_base[i].flags.bits.lb
#define ILT_DBGLINE(i) iltb.stg_base[i].flags.bits.dbgline
#define ILT_DELETE(i) iltb.stg_base[i].flags.bits.delete_
#define ILT_IGNORE(i) iltb.stg_base[i].flags.bits.ignore
#define ILT_SPLIT(i) iltb.stg_base[i].flags.bits.split
#define ILT_CPLX(i) iltb.stg_base[i].flags.bits.cplx
#define ILT_MCACHE(i) iltb.stg_base[i].flags.bits.mcache
#define ILT_NODEL(i) iltb.stg_base[i].flags.bits.nodel
#define ILT_KEEP(i) iltb.stg_base[i].flags.bits.keep
#define ILT_DELEBB(i) iltb.stg_base[i].flags.bits.delebb
#define ILT_PREDC(i) iltb.stg_base[i].flags.bits.predc
#define ILT_EQASRT(i) iltb.stg_base[i].flags.bits.eqasrt
#define ILT_FREE(i) iltb.stg_base[i].flags.bits.free
#define ILT_INV(i) iltb.stg_base[i].flags.bits.inv

/*****  ILT External Data Declarations *****/

extern ILTB iltb;

#include "iltutil.h"

#endif // ILT_H_
