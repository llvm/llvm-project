/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*  induction.h - Global data structures for induction analysis */

#include <stdint.h>

/*****  Induction Use Item for an induction variable  *****/

typedef struct IUSE_T {
  struct IUSE_T *next; /* next induction use for the variable */
  int ast;             /* ast using the induction var, <i> */
  int use;             /* use entry for the induction var */
  char type;           /* type of use:
                        *   '+', '-', '*',
                        *   'm' - register move
                        *   'b' - branch   'p' - pseudo store
                        */
  char opn;            /* operand # of <x> in <i> <op> <x>
                        * (0, if 1 operand)
                        * (3, ==> use is combines two ind variables)
                        */
  char sclfg;          /* use (AADD or ASUB) has a scale */
} IUSE;

/*****  Induction Table Entry  *****/

typedef struct {
  int nm;     /* names entry of induction variable */
  int load;   /* ast of the load of the induction var */
  int init;   /* initial value of basic induction var:
               * will either be the load ast or the value
               * of a single def which reaches this head
               * of the loop.
               */
  int family; /* "family of" (ind index) */
  union {
    uint16_t all;
    struct {
      uint16_t omit : 1;   /* names not an induction variable */
      uint16_t ptr : 1;    /* induction variable is a pointer */
      uint16_t delete : 1; /* induction variable can be deleted */
      uint16_t niu : 1;    /* iv has non-induction use */
      uint16_t gone : 1;   /* iv has been deleted */
      uint16_t midf : 1;   /* multiple defs for iv */
      uint16_t rmst : 1;   /* found by removest (def in nested blocks) */
      uint16_t alias : 1;  /* var is an induction alias of "family" */
    } bits;
  } flags;
  INT16 opc;    /* ast opcode of skip expr */
  int skip;     /* ast of skip (0 if skips are not the same)*/
  int mult;     /* the multiplier (an ast) which is applied to
                 * iv's "family of" variable.
                 */
  int initdef;  /* def entry if there exists a def which
                 * provides biv's initial value; 0 if it
                 * doesn't exist.
                 */
  Q_ITEM *bivl; /* list of basic induction definitions -
                 * The info field is the def pointer.
                 * The flag field has the format:
                 *      15       11       7              0
                 *     +--------+--------+----------------+
                 *     |IDT_type|  opn#  |   '+' or '-'   |
                 *     +--------+--------+----------------+
                 * NOTE:
                 *   when an induction definition is deleted
                 *   its corresponding info field is relaced
                 *   with the def's store ast.
                 */
  IUSE *usel;   /* induction use list;  first item is actually
                 * the list head.  Its use field is the number
                 * of pseudo store uses.
                 */
  Q_ITEM *astl; /* list of ast assigned to this variable */
  int derived;  /* the induction variable (ind index) from
                 * which this induction variable is derived
                 */
} IND;

#define IND_NM(i) induc.indb.stg_base[i].nm
#define IND_LOAD(i) induc.indb.stg_base[i].load
#define IND_INIT(i) induc.indb.stg_base[i].init
#define IND_FAMILY(i) induc.indb.stg_base[i].family
#define IND_FLAGS(i) induc.indb.stg_base[i].flags.all
#define IND_OMIT(i) induc.indb.stg_base[i].flags.bits.omit
#define IND_PTR(i) induc.indb.stg_base[i].flags.bits.ptr
#define IND_DELETE(i) induc.indb.stg_base[i].flags.bits.delete
#define IND_NIU(i) induc.indb.stg_base[i].flags.bits.niu
#define IND_GONE(i) induc.indb.stg_base[i].flags.bits.gone
#define IND_MIDF(i) induc.indb.stg_base[i].flags.bits.midf
#define IND_RMST(i) induc.indb.stg_base[i].flags.bits.rmst
#define IND_ALIAS(i) induc.indb.stg_base[i].flags.bits.alias
#define IND_OPC(i) induc.indb.stg_base[i].opc
#define IND_SKIP(i) induc.indb.stg_base[i].skip
#define IND_BIVL(i) induc.indb.stg_base[i].bivl
#define IND_USEL(i) induc.indb.stg_base[i].usel
#define IND_ASTL(i) induc.indb.stg_base[i].astl
#define IND_MULT(i) induc.indb.stg_base[i].mult
#define IND_DERIVED(i) induc.indb.stg_base[i].derived
#define IND_INITDEF(i) induc.indb.stg_base[i].initdef
#define BIVL_IDT_MSK 0x3
#define BIVL_IDT_SHF 12
#define BIVL_OPN_MSK 0x3
#define BIVL_OPN_SHF 8
#define BIVL_OPC_MSK 0xff

#define GET_DU(du)                             \
  {                                            \
    if (induc.mark_du == NULL)                 \
      du = (DU *)getitem(DU_AREA, sizeof(DU)); \
    else {                                     \
      du = induc.mark_du;                      \
      induc.mark_du = induc.mark_du->next;     \
    }                                          \
  }

/*****  Induction Common  *****/

typedef struct {
  struct {/* storage allocation for induction table */
    IND *stg_base;
    int stg_size;
    int stg_avail;
  } indb;
  DU *mark_du;    /* list of du items to re-use */
  int last_biv;   /* ind entry of last basic induction variable */
  int branch_ind; /* ind of biv used in replaceable branch */
} INDUC;

extern INDUC induc;
