//===-- smoke tests for RPC -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc.h"

#include "test/UnitTest/Test.h"

namespace {
enum { lane_size = 8, port_count = 4 };

struct Packet {
  uint64_t unused;
};

using ProcAType = __llvm_libc::rpc::Process<false, Packet>;
using ProcBType = __llvm_libc::rpc::Process<true, Packet>;

static_assert(ProcAType::inbox_offset(port_count) ==
              ProcBType::outbox_offset(port_count));

static_assert(ProcAType::outbox_offset(port_count) ==
              ProcBType::inbox_offset(port_count));

enum { alloc_size = ProcAType::allocation_size(port_count) };

alignas(64) char buffer[alloc_size] = {0};
} // namespace

TEST(LlvmLibcRPCSmoke, SanityCheck) {

  ProcAType ProcA(port_count, buffer);
  ProcBType ProcB(port_count, buffer);

  uint64_t index = 0; // any < port_count
  uint64_t lane_mask = 1;

  // Each process has its own local lock for index
  EXPECT_TRUE(ProcA.try_lock(lane_mask, index));
  EXPECT_TRUE(ProcB.try_lock(lane_mask, index));

  // All zero to begin with
  EXPECT_EQ(ProcA.load_inbox(lane_mask, index), 0u);
  EXPECT_EQ(ProcB.load_inbox(lane_mask, index), 0u);
  EXPECT_EQ(ProcA.load_outbox(lane_mask, index), 0u);
  EXPECT_EQ(ProcB.load_outbox(lane_mask, index), 0u);

  // Available for ProcA and not for ProcB
  EXPECT_FALSE(ProcA.buffer_unavailable(ProcA.load_inbox(lane_mask, index),
                                        ProcA.load_outbox(lane_mask, index)));
  EXPECT_TRUE(ProcB.buffer_unavailable(ProcB.load_inbox(lane_mask, index),
                                       ProcB.load_outbox(lane_mask, index)));

  // ProcA write to outbox
  uint32_t ProcAOutbox = ProcA.load_outbox(lane_mask, index);
  EXPECT_EQ(ProcAOutbox, 0u);
  ProcAOutbox = ProcA.invert_outbox(index, ProcAOutbox);
  EXPECT_EQ(ProcAOutbox, 1u);

  // No longer available for ProcA
  EXPECT_TRUE(ProcA.buffer_unavailable(ProcA.load_inbox(lane_mask, index),
                                       ProcAOutbox));

  // Outbox is still zero, hasn't been written to
  EXPECT_EQ(ProcB.load_outbox(lane_mask, index), 0u);

  // Wait for ownership will terminate because load_inbox returns 1
  EXPECT_EQ(ProcB.load_inbox(lane_mask, index), 1u);
  ProcB.wait_for_ownership(lane_mask, index, 0u, 0u);

  // and B now has the buffer available
  EXPECT_FALSE(ProcB.buffer_unavailable(ProcB.load_inbox(lane_mask, index),
                                        ProcB.load_outbox(lane_mask, index)));

  // Enough checks for one test, close the locks
  ProcA.unlock(lane_mask, index);
  ProcB.unlock(lane_mask, index);
}
