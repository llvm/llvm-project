//===-- sanitizer_chained_origin_depot.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A storage for chained origins.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_CHAINED_ORIGIN_DEPOT_H
#define SANITIZER_CHAINED_ORIGIN_DEPOT_H

#include "sanitizer_common.h"
#include "sanitizer_stackdepotbase.h"

namespace __sanitizer {

class ChainedOriginDepot {
 public:
  ChainedOriginDepot();

  // Gets the statistic of the origin chain storage.
  StackDepotStats *GetStats();

  // Stores a chain with StackDepot ID here_id and previous chain ID prev_id.
  // If successful, returns true and the new chain id new_id.
  // If the same element already exists, returns false and sets new_id to the
  // existing ID.
  bool Put(u32 here_id, u32 prev_id, u32 *new_id);

  // Retrieves the stored StackDepot ID for the given origin ID.
  u32 Get(u32 id, u32 *other);

  void LockAll();
  void UnlockAll();

 private:
  struct ChainedOriginDepotDesc {
    u32 here_id;
    u32 prev_id;
  };

  struct ChainedOriginDepotNode {
    ChainedOriginDepotNode *link;
    u32 id;
    u32 here_id;
    u32 prev_id;

    typedef ChainedOriginDepotDesc args_type;

    bool eq(u32 hash, const args_type &args) const;

    static uptr storage_size(const args_type &args);

    static u32 hash(const args_type &args);

    static bool is_valid(const args_type &args);

    void store(const args_type &args, u32 other_hash);

    args_type load() const;

    struct Handle {
      ChainedOriginDepotNode *node_;
      Handle() : node_(nullptr) {}
      explicit Handle(ChainedOriginDepotNode *node) : node_(node) {}
      bool valid() { return node_; }
      u32 id() { return node_->id; }
      int here_id() { return node_->here_id; }
      int prev_id() { return node_->prev_id; }
    };

    Handle get_handle();

    typedef Handle handle_type;
  };

  StackDepotBase<ChainedOriginDepotNode, 4, 20> depot;

  ChainedOriginDepot(const ChainedOriginDepot &) = delete;
  void operator=(const ChainedOriginDepot &) = delete;
};

}  // namespace __sanitizer

#endif  // SANITIZER_CHAINED_ORIGIN_DEPOT_H
