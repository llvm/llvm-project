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
void __trec_branch(u64 cond) { CondBranch(cur_thread(), CALLERPC, cond); }

void __trec_func_param(u16 param_idx, void *src_addr, u16 src_idx, void *val) {
  FuncParam(cur_thread(), param_idx, (uptr)src_addr, src_idx, (uptr)val);
}

void __trec_func_exit_param(void *src_addr, u16 src_idx, void *val) {
  FuncExitParam(cur_thread(), (uptr)src_addr, src_idx, (uptr)val);
}

void __trec_inst_debug_info(u32 line, u16 col, char *val_name,
                            char *addr_name) {
  if (LIKELY(ctx->flags.output_trace) && LIKELY(ctx->flags.output_debug) &&
      LIKELY(cur_thread()->ignore_interceptors == 0))
    if ((ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3)) {
      __trec_debug_info::InstDebugInfo info(
          line, col,
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

void __trec_read1(void *addr, bool isPtr, void *val, void *addr_src_addr,
                  u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog1, isPtr, (uptr)val,
             SAI_addr);
}

void __trec_read2(void *addr, bool isPtr, void *val, void *addr_src_addr,
                  u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog2, isPtr, (uptr)val,
             SAI_addr);
}

void __trec_read4(void *addr, bool isPtr, void *val, void *addr_src_addr,
                  u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog4, isPtr, (uptr)val,
             SAI_addr);
}

void __trec_read8(void *addr, bool isPtr, void *val, void *addr_src_addr,
                  u16 addr_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  MemoryRead(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8, isPtr, (uptr)val,
             SAI_addr);
}

void __trec_write1(void *addr, bool isPtr, void *val, void *addr_src_addr,
                   u16 addr_src_idx, void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog1, isPtr, (uptr)val,
              SAI_addr, SAI_val);
}

void __trec_write2(void *addr, bool isPtr, void *val, void *addr_src_addr,
                   u16 addr_src_idx, void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog2, isPtr, (uptr)val,
              SAI_addr, SAI_val);
}

void __trec_write4(void *addr, bool isPtr, void *val, void *addr_src_addr,
                   u16 addr_src_idx, void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog4, isPtr, (uptr)val,
              SAI_addr, SAI_val);
}

void __trec_write8(void *addr, bool isPtr, void *val, void *addr_src_addr,
                   u16 addr_src_idx, void *val_src_addr, u16 val_src_idx) {
  SourceAddressInfo SAI_addr(addr_src_idx, (uptr)addr_src_addr);
  SourceAddressInfo SAI_val(val_src_idx, (uptr)val_src_addr);
  MemoryWrite(cur_thread(), CALLERPC, (uptr)addr, kSizeLog8, isPtr, (uptr)val,
              SAI_addr, SAI_val);
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
