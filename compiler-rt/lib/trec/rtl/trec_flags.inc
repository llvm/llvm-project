//===-- trec_flags.inc ------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TRec runtime flags.
//
//===----------------------------------------------------------------------===//
#ifndef TREC_FLAG
#error "Define TREC_FLAG prior to including this file!"
#endif

// TREC_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

TREC_FLAG(
    bool, force_seq_cst_atomics, false,
    "If set, all atomics are effectively sequentially consistent (seq_cst), "
    "regardless of what user actually specified.")
TREC_FLAG(int, atexit_sleep_ms, 0,
               "Sleep in main thread before exiting for that many ms "
               "(useful to catch \"at exit\" races).")
TREC_FLAG(
    bool, die_after_fork, true,
    "Die after multi-threaded fork if the child creates new threads.")
TREC_FLAG(bool, output_trace, true, "Output trace.")
TREC_FLAG(bool, record_mutex, true, "Record Mutex Lock/Unlock Operations?")
TREC_FLAG(bool, record_alloc_free, true, "Record Memory Object Allocate/Free?")
TREC_FLAG(bool, no_stack, false, "Do not record memory operation on stack.")
TREC_FLAG(bool, record_branch, true, "Record conditional branch insts.")
TREC_FLAG(bool, record_func_enter_exit, true, "Record function enter/exit insts.")
TREC_FLAG(bool, record_func_param, true, "Record function parameter insts.")
TREC_FLAG(int, trace_mode, 2, "Trace Mode: 1. SeqCheck, 2. Eagle, 3. Verification")\
TREC_FLAG(bool, record_read, true, "Record Memory Read Operations.")
TREC_FLAG(bool, record_write,true,"Record Memory Write Operations." )
TREC_FLAG(bool, output_debug, true, "Record Debug Information.")
