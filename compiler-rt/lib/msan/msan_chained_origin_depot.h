//===-- msan_chained_origin_depot.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A storage for chained origins.
//===----------------------------------------------------------------------===//
#ifndef MSAN_CHAINED_ORIGIN_DEPOT_H
#define MSAN_CHAINED_ORIGIN_DEPOT_H

#include "sanitizer_common/sanitizer_common.h"

namespace __msan {

StackDepotStats *ChainedOriginDepotGetStats();
bool ChainedOriginDepotPut(u32 here_id, u32 prev_id, u32 *new_id);
// Retrieves a stored stack trace by the id.
u32 ChainedOriginDepotGet(u32 id, u32 *other);

void ChainedOriginDepotLockAll();
void ChainedOriginDepotUnlockAll();

}  // namespace __msan

#endif  // MSAN_CHAINED_ORIGIN_DEPOT_H
