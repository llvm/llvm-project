/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/**
 *  \file
 *  \brief - publicly accessible functions for inliner
 */
extern void extractor_command_info(char *sDir, int ignore, char *sFunc);
extern void extractor_end(void);
extern void extractor(void);
extern int extractor_possible(void);
extern void inline_add_lib(char *sDir);
extern void inline_add_func(char *sFunc, int nSize);
extern void inliner(void);
