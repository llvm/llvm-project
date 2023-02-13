//===-- trec_rtl_proc.cpp -----------------------------------------------===//
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

#include "sanitizer_common/sanitizer_placement_new.h"
#include "trec_rtl.h"
#include "trec_mman.h"
#include "trec_flags.h"

namespace __trec {

Processor *ProcCreate() {
  void *mem = InternalAlloc(sizeof(Processor));
  internal_memset(mem, 0, sizeof(Processor));
  Processor *proc = new(mem) Processor;
  proc->thr = nullptr;
#if !SANITIZER_GO
  AllocatorProcStart(proc);
#endif
  return proc;
}

void ProcDestroy(Processor *proc) {
  CHECK_EQ(proc->thr, nullptr);
#if !SANITIZER_GO
  AllocatorProcFinish(proc);
#endif
  proc->~Processor();
  InternalFree(proc);
}

void ProcWire(Processor *proc, ThreadState *thr) {
  CHECK_EQ(thr->proc1, nullptr);
  CHECK_EQ(proc->thr, nullptr);
  thr->proc1 = proc;
  proc->thr = thr;
}

void ProcUnwire(Processor *proc, ThreadState *thr) {
  CHECK_EQ(thr->proc1, proc);
  CHECK_EQ(proc->thr, thr);
  thr->proc1 = nullptr;
  proc->thr = nullptr;
}

}  // namespace __trec
