/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* mp.h  -  various OpenMP definitions */

#ifndef __MP_H__
#define __MP_H__

/* Bit-maskable OpenMP Tasking Flags */
#define MP_TASK_UNTIED 0x01
#define MP_TASK_IF 0x02
#define MP_TASK_ORPHANED 0x04
#define MP_TASK_NESTED 0x08
#define MP_TASK_FORCED_DEFER 0x10
#define MP_TASK_FINAL 0x20
#define MP_TASK_IMMEDIATE 0x40
#define MP_TASK_MERGEABLE 0x80
#define MP_TASK_PRIORITY 0x100
#define MP_TASK_NOGROUP 0x1000
#define MP_TASK_GRAINSIZE 0x2000
#define MP_TASK_NUM_TASKS 0x4000

/* Schedule attributes for MP_SCH_
 * These are used to represent the MP_XXX for C or DI_XXX for FTN
 *
 * Basic type of schedule (auto, static, dynamic, guided, etc.)
 * are represented by the low byte
 */
#define MP_SCH_TYPE_MASK 0x000000FF
#define MP_SCH_STATIC 0x0
#define MP_SCH_DYNAMIC 0x1
#define MP_SCH_GUIDED 0x2
#define MP_SCH_INTERLEAVE 0x3
#define MP_SCH_RUNTIME 0x4
#define MP_SCH_AUTO 0x5
#define MP_SCH_DIST_STATIC 0x6  /* use in distribute parallel for */
#define MP_SCH_DIST_DYNAMIC 0x7 /* use in distribute parallel for */

/* The second byte represents special case flags for static (maskable) */
#define MP_SCH_SPC_MASK 0x0000FF00
#define MP_SCH_SPC_SHIFT 8
#define MP_SCH_CHUNK_1 0x00000100 /* Chunk == 1 (static cyclic) */
#define MP_SCH_BLK_CYC 0x00000200 /* Chunk > 1  (block cyclic)  */
#define MP_SCH_BLK_ALN 0x00000400 /* Static block aligned       */

/* The high (third) byte represents attributes (maskable) */
#define MP_SCH_ATTR_MASK 0x00FF0000
#define MP_SCH_ATTR_SHIFT 16
#define MP_SCH_ATTR_ORDERED 0x00010000 /* Ordered */
#define MP_SCH_ATTR_CHUNKED 0x00020000 /* Chunked */
#define MP_SCH_ATTR_DIST 0x00040000    /* distributed */
#define MP_SCH_ATTR_DEVICEDIST 0x00080000    /* fast GPU scheduler for TTDPF */

/* Target/Target combine attribute */
#define MP_TGT_NOWAIT 0x01   /* if NOWAIT is present */
#define MP_TGT_IFTARGET 0x02 /* IF(target)   clause is present */
#define MP_TGT_IFPAR 0x04    /* IF(parallel) clause is present */
#define MP_TGT_DEPEND_IN                               \
  0x08 /* depend is present and has dependence-type IN \
          */
#define MP_TGT_DEPEND_OUT \
  0x10 /* depend is present and has dependence-type OUT */
#define MP_TGT_DEPEND_INOUT \
  0x20 /* Depend is present and has dependence-type INOUT */
#define MP_CMB_TEAMS      0x40  /* teams clause is present */
#define MP_CMB_DISTRIBUTE 0x80  /* distribute clause is present */
#define MP_CMB_PARALLEL   0x100 /* parallel clause is present */
#define MP_CMB_FOR        0x200 /* for clause is present */
#define MP_CMB_SIMD       0x400 /* simd clause is present */
#define MP_CMB_PARFOR     (MP_CMB_FOR|MP_CMB_PARALLEL)

typedef enum omp_proc_bind_t {
  MP_PROC_BIND_FALSE = 0,
  MP_PROC_BIND_TRUE,
  MP_PROC_BIND_MASTER,
  MP_PROC_BIND_CLOSE,
  MP_PROC_BIND_SPREAD,
} omp_proc_bind_t;

typedef enum omp_iftype {
  IF_DEFAULT = 0,
  IF_TARGET = 1,
  IF_TARGETDATA = (1 << 1),
  IF_TARGETENTERDATA = (1 << 2),
  IF_TARGETEXITDATA = (1 << 3),
  IF_TARGETUPDATE = (1 << 4),
  IF_PARALLEL = (1 << 5),
  IF_TASK = (1 << 6),
  IF_TASKLOOP = (1 << 7),
} omp_iftype;

/* Keep up to date with pgcplus_omp_cancel_type init_omp()*/
typedef enum omp_canceltype {
  CANCEL_PARALLEL = 1,
  CANCEL_FOR = 2,
  CANCEL_SECTIONS = 3,
  CANCEL_TASKGROUP = 4,
} omp_canceltype;

#endif /* __MP_H__ */
