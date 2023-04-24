//===-- trec_interface_inl.h ------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_ptrauth.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "trec_interface.h"
#include "trec_rtl.h"

#define CALLERPC \
  (StackTrace::GetPreviousInstructionPc((uptr)__builtin_return_address(0)))

using namespace __trec;
using namespace __trec_metadata;

void __trec_inst_debug_info(u64 fid, u32 line, u16 col, u64 time,
                            char *val_name, char *addr_name) {
  if (LIKELY(ctx->flags.output_trace) && LIKELY(ctx->flags.output_debug) &&
      LIKELY(cur_thread()->ignore_interceptors == 0))
    if ((ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3)) {
      __trec_debug_info::InstDebugInfo info(
          fid, line, col, time,
          min(internal_strlen(val_name),
              (uptr)((1 << (8 * sizeof(info.name_len[0]))) - 1)),
          min(internal_strlen(addr_name),
              (uptr)((1 << (8 * sizeof(info.name_len[1]))) - 1)));
      ThreadState *thr = cur_thread();

      internal_memcpy(thr->tctx->dbg_temp_buffer, &info, sizeof(info));
      internal_memcpy(thr->tctx->dbg_temp_buffer + sizeof(info), val_name,
                      info.name_len[0]);
      internal_memcpy(
          thr->tctx->dbg_temp_buffer + sizeof(info) + info.name_len[0],
          addr_name, info.name_len[1]);
      thr->tctx->dbg_temp_buffer_size =
          sizeof(info) + info.name_len[0] + info.name_len[1];
    }
}

void __trec_func_entry(void *name) {
  bool should_record = true;
  RecordFuncEntry(cur_thread(), should_record, (char *)name,
                  StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}

void __trec_func_exit() {
  bool should_record = true;
  RecordFuncExit(cur_thread(), should_record, __func__);
}

void __trec_bbl_entry() {
  bool should_record = true;
  RecordBBLEntry(cur_thread(), should_record);
}

