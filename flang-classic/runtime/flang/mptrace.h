/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/** \file
 * external declarations for libpgc routinens in mptrace.c
 */

extern void _mp_trace_parallel_enter(void);
extern void _mp_trace_parallel_exit(void);
extern void _mp_trace_parallel_begin(void);
extern void _mp_trace_parallel_end(void);
extern void _mp_trace_sections_enter(void);
extern void _mp_trace_sections_exit(void);
extern void _mp_trace_section_begin(void);
extern void _mp_trace_section_end(void);
extern void _mp_trace_single_enter(void);
extern void _mp_trace_single_exit(void);
extern void _mp_trace_master_enter(void);
extern void _mp_trace_master_exit(void);
extern void _mp_trace_loop_enter(void);
extern void _mp_trace_loop_exit(void);


