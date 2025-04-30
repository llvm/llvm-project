/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * external declarations for libpgc routines defined in llcrit.c 
 */

#include "komp.h"

extern void _mp_p(kmp_critical_name *sem);
extern void _mp_v(kmp_critical_name *sem);
extern void _mp_ptest(omp_lock_t *sem);
extern void _mp_bcs(void);
extern void _mp_ecs(void);
extern void _mp_bcs_stdio(void);
extern void _mp_ecs_stdio(void);
extern void _mp_bcs_nest(void);
extern void _mp_ecs_nest(void);
extern void _mp_cdeclp(void *blk, void ***blk_tp, int size);
extern void _mp_cdecli(void *blk, void ***blk_tp, int size);
extern void _mp_cdecl(void *blk, void ***blk_tp, int size);
extern void _mp_copyin_init(void);
extern void _mp_copypriv_init(void);
extern void _mp_copyin_term(void);
extern void _mp_copypriv_term(void);
extern void _mp_copypriv(char *adr, long len, int thread);
extern void _mp_copypriv_al(char **adr, long len, int thread);
extern void _mp_copypriv_move(void *blk_tp, int off, int size,
                              int single_thread);
extern void _mp_copyin_move(void *blk_tp, int off, int size);
extern void _mp_copyin_move_al(void *blk_tp, int off, long size);

#define MP_P(sem) _mp_p(&sem)
#define MP_V(sem) _mp_v(&sem)
#define MP_PTEST(sem) _mp_ptest(sem)
#define MP_BCS _mp_bcs()
#define MP_ECS _mp_ecs()
#define MP_P_STDIO _mp_bcs_stdio()
#define MP_V_STDIO _mp_ecs_stdio()
#define MP_BCS_NEST _mp_bcs_nest()
#define MP_ECS_NEST _mp_ecs_nest()
#define MP_CDECLP(blk, blk_tp, size) _mp_cdeclp(blk, blk_tp, size)
#define MP_CDECLI(blk, blk_tp, size) _mp_cdecli(blk, blk_tp, size)
#define MP_CDECL(blk, blk_tp, size) _mp_cdecl(blk, blk_tp, size)
#define MP_COPYIN_INIT _mp_copyin_init()
#define MP_COPYPRIV_INIT _mp_copypriv_init()
#define MP_COPYIN_TERM _mp_copyin_term()
#define MP_COPYPRIV_TERM _mp_copypriv_term()
#define MP_COPYPRIV(adr, len, thread) _mp_copypriv(adr, len, thread)
#define MP_COPYPRIV_AL(adr, len, thread) _mp_copypriv_al(adr, len, thread)
#define MP_COPYPRIV_MOVE(blk_tp, off, size, single_thread)  \
          _mp_copypriv_move(blk_tp, off, size, single_thread)
#define MP_COPYIN_MOVE(blk_tp, off, size) _mp_copyin_move(blk_tp, off, size)
#define MP_COPYIN_MOVE_AL(blk_tp, off, size) _mp_copyin_move_al(blk_tp, off, size)

#define MP_SEMAPHORE(sc, sem) sc kmp_critical_name sem;
