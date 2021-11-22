//===-- memprof_mibmap.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
//===----------------------------------------------------------------------===//

#include "memprof_mibmap.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __memprof {

void InsertOrMerge(const uptr Id, const MemInfoBlock &Block, MIBMapTy &Map) {
  MIBMapTy::Handle h(&Map, static_cast<uptr>(Id), /*remove=*/false,
                     /*create=*/true);
  if (h.created()) {
    LockedMemInfoBlock *lmib =
        (LockedMemInfoBlock *)InternalAlloc(sizeof(LockedMemInfoBlock));
    lmib->mutex.Init();
    lmib->mib = Block;
    *h = lmib;
  } else {
    LockedMemInfoBlock *lmib = *h;
    SpinMutexLock lock(&lmib->mutex);
    lmib->mib.Merge(Block);
  }
}

} // namespace __memprof
