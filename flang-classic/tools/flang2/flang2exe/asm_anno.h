/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef ASM_ANNO_H_
#define ASM_ANNO_H_

/**
   \file
   \brief Interface to annotation module

   Define interface to Annotation module: which supports assembly file
   annotation (-Manno switch).

   Annotation is supported by the following functions - generally called from
   the Scheduler/Code Generator module, iff "flg.anno" is true.
 */

/**
   \brief ...
 */
void anno_blkcnt(int blkno);

/**
   \brief should be called before emitting assembly code for each basic block
 */
void annomod_asm(int blkno);

/**
   \brief should be called after all code is emitted for the current user
   function, but before assem_end_func() is called.
 */
void annomod_end(void);

/**
   \brief should be called at the beginning of processing for each user function
   (but not before merge_blocks() is called).
 */
void annomod_init(void);

#endif
