//===-- sanitizer_chained_origin_depot.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A storage for chained origins.
//===----------------------------------------------------------------------===//

#include "sanitizer_chained_origin_depot.h"

namespace __sanitizer {

bool ChainedOriginDepot::ChainedOriginDepotNode::eq(
    u32 hash, const args_type &args) const {
  return here_id == args.here_id && prev_id == args.prev_id;
}

uptr ChainedOriginDepot::ChainedOriginDepotNode::storage_size(
    const args_type &args) {
  return sizeof(ChainedOriginDepotNode);
}

/* This is murmur2 hash for the 64->32 bit case.
   It does not behave all that well because the keys have a very biased
   distribution (I've seen 7-element buckets with the table only 14% full).

   here_id is built of
   * (1 bits) Reserved, zero.
   * (8 bits) Part id = bits 13..20 of the hash value of here_id's key.
   * (23 bits) Sequential number (each part has each own sequence).

   prev_id has either the same distribution as here_id (but with 3:8:21)
   split, or one of two reserved values (-1) or (-2). Either case can
   dominate depending on the workload.
*/
u32 ChainedOriginDepot::ChainedOriginDepotNode::hash(const args_type &args) {
  const u32 m = 0x5bd1e995;
  const u32 seed = 0x9747b28c;
  const u32 r = 24;
  u32 h = seed;
  u32 k = args.here_id;
  k *= m;
  k ^= k >> r;
  k *= m;
  h *= m;
  h ^= k;

  k = args.prev_id;
  k *= m;
  k ^= k >> r;
  k *= m;
  h *= m;
  h ^= k;

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}

bool ChainedOriginDepot::ChainedOriginDepotNode::is_valid(
    const args_type &args) {
  return true;
}

void ChainedOriginDepot::ChainedOriginDepotNode::store(const args_type &args,
                                                       u32 other_hash) {
  here_id = args.here_id;
  prev_id = args.prev_id;
}

ChainedOriginDepot::ChainedOriginDepotNode::args_type
ChainedOriginDepot::ChainedOriginDepotNode::load() const {
  args_type ret = {here_id, prev_id};
  return ret;
}

ChainedOriginDepot::ChainedOriginDepotNode::Handle
ChainedOriginDepot::ChainedOriginDepotNode::get_handle() {
  return Handle(this);
}

ChainedOriginDepot::ChainedOriginDepot() {}

StackDepotStats *ChainedOriginDepot::GetStats() { return depot.GetStats(); }

bool ChainedOriginDepot::Put(u32 here_id, u32 prev_id, u32 *new_id) {
  ChainedOriginDepotDesc desc = {here_id, prev_id};
  bool inserted;
  ChainedOriginDepotNode::Handle h = depot.Put(desc, &inserted);
  *new_id = h.valid() ? h.id() : 0;
  return inserted;
}

u32 ChainedOriginDepot::Get(u32 id, u32 *other) {
  ChainedOriginDepotDesc desc = depot.Get(id);
  *other = desc.prev_id;
  return desc.here_id;
}

void ChainedOriginDepot::LockAll() { depot.LockAll(); }

void ChainedOriginDepot::UnlockAll() { depot.UnlockAll(); }

}  // namespace __sanitizer
