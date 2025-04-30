/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief External declarations for memory management routines defined in alloc.c.
 */

void ENTF90(PTR_ALLOC03,
            ptr_alloc03)(__INT_T *nelem, __INT_T *kind, __INT_T *len,
                         __STAT_T *stat, char **pointer, __POINT_T *offset,
                         __INT_T *firsttime, DCHAR(errmsg) DCLEN(errmsg));

void ENTF90(PTR_SRC_ALLOC03,
            ptr_src_alloc03)(F90_Desc *sd, __INT_T *nelem, __INT_T *kind,
                             __INT_T *len, __STAT_T *stat, char **pointer,
                             __POINT_T *offset, __INT_T *firsttime,
                             DCHAR(errmsg) DCLEN(errmsg));

void ENTF90(DEALLOC03, dealloc03)(__STAT_T *stat, char *area,
                                  __INT_T *firsttime,
                                  DCHAR(errmsg) DCLEN(errmsg));

void ENTF90(DEALLOC03A, dealloc03a)(__STAT_T *stat, char *area,
                                    __INT_T *firsttime,
                                    DCHAR(errmsg) DCLEN64(errmsg));

void ENTF90(DEALLOC_MBR03, dealloc_mbr03)(__STAT_T *stat, char *area,
                                          __INT_T *firsttime,
                                          DCHAR(errmsg) DCLEN(errmsg));
