/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "lz.h"

typedef union {
  struct {
    int descr;
    int rank;
    int flag;
    int dist_target;
    int dist_target_descr;
    int isstar;
    int align_target;
    int target_descr;
    int conform;
    int collapse;
    int lb[MAXRANK];
    int ub[MAXRANK];
    int type;
    int alignee_sc;
    int target_sc;
  } template;
  struct {
    int descr;
    int rank;
    int template;
    int dtype;
    int kind;
    int size;
  } instance;
  int which; /* 1 for template, 2 for instance */
} DTABLE;

typedef struct {
  DTABLE *base;
  int size;
  int avl;
} DTB;

extern DTB dtb;

#define TMPL_DESCR(i) dtb.base[i].template.descr
#define TMPL_RANK(i) dtb.base[i].template.rank
#define TMPL_FLAG(i) dtb.base[i].template.flag
#define TMPL_DIST_TARGET(i) dtb.base[i].template.dist_target
#define TMPL_DIST_TARGET_DESCR(i) dtb.base[i].template.dist_target_descr
#define TMPL_ISSTAR(i) dtb.base[i].template.isstar
#define TMPL_ALIGN_TARGET(i) dtb.base[i].template.align_target
#define TMPL_TARGET_DESCR(i) dtb.base[i].template.target_descr
#define TMPL_CONFORM(i) dtb.base[i].template.conform
#define TMPL_COLLAPSE(i) dtb.base[i].template.collapse
#define TMPL_LB(i, j) dtb.base[i].template.lb[j]
#define TMPL_UB(i, j) dtb.base[i].template.ub[j]
#define TMPL_TYPE(i) dtb.base[i].template.type
#define TMPL_ALIGNEE_SC(i) dtb.base[i].template.alignee_sc
#define TMPL_TARGET_SC(i) dtb.base[i].template.target_sc

#define INS_DESCR(i) dtb.base[i].instance.descr
#define INS_RANK(i) dtb.base[i].instance.rank
#define INS_TEMPLATE(i) dtb.base[i].instance.template
#define INS_DTYPE(i) dtb.base[i].instance.dtype
#define INS_KIND(i) dtb.base[i].instance.kind
#define INS_SIZE(i) dtb.base[i].instance.size

#define __ASSUMED_SIZE 0x0001
#define __SEQUENTIAL 0x0002
#define __ASSUMED_SHAPE 0x0004
#define __SAVE 0x0008

#define __INHERIT 0x0010
#define __NO_OVERLAPS 0x00020

#define __INTENT_INOUT 0
#define __INTENT_IN 0x0040
#define __INTENT_OUT 0x0080

#define __OMITTED_DIST_TARGET 0
#define __PRESCRIPTIVE_DIST_TARGET 0x0100
#define __DESCRIPTIVE_DIST_TARGET 0x0200
#define __TRANSCRIPTIVE_DIST_TARGET 0x0300

#define __OMITTED_DIST_FORMAT 0
#define __PRESCRIPTIVE_DIST_FORMAT 0x0400
#define __DESCRIPTIVE_DIST_FORMAT 0x0800
#define __TRANSCRIPTIVE_DIST_FORMAT 0x0c00

#define __PRESCRIPTIVE_ALIGN_TARGET 0x1000
#define __DESCRIPTIVE_ALIGN_TARGET 0x2000
#define __IDENTITY_MAP 0x4000

#define __DYNAMIC 0x8000
#define __POINTER 0x10000
#define __LOCAL 0x20000
#define __F77_LOCAL_DUMMY 0x40000
#define __OFF_TEMPLATE 0x80000

#define REPLICATED 0
#define DISTRIBUTED 1
#define ALIGNED 2
#define INHERITED 3

#define DIST_TARGET_SHIFT 8
#define DIST_FORMAT_SHIFT 10
#define ALIGN_TARGET_SHIFT 12

#define OMIT 0
#define PRESCRIP 1
#define DESCRIP 2
#define TRANSCRIP 3

#define NONE_SC 0
#define ALLOC_SC 1
#define DUMMY_SC 2
#define STATIC_SC 3
#define COMMON_SC 4

typedef struct {
  int *base;
  int avl;
  int size;
} FL;

extern FL fl;

void set_typed_alloc(DTYPE);
void set_type_in_descriptor(int descriptor_ast, int sptr, DTYPE dtype,
                            int parent_ast, int before_std);
int make_simple_template_from_ast(int ast, int std, LOGICAL need_type_in_descr);

int newargs_for_llvmiface(int sptr);
void interface_for_llvmiface(int this_entry, int new_dscptr);
void undouble_callee_args_llvmf90(int iface);
