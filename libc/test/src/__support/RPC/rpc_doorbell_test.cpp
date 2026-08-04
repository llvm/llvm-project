//===-- tests for the RPC doorbell ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/RPC/rpc.h"

#include "test/UnitTest/Test.h"

namespace {
enum { port_count = 4 };

using ProcType = LIBC_NAMESPACE::rpc::Process<false>;

enum { alloc_size = ProcType::allocation_size(port_count, 1) };

alignas(64) char buffer[alloc_size] = {0};

// At namespace scope so the doorbell never refers to a dead local.
uint64_t pending = 0;
} // namespace

// A doorbell can be only partially initialized. The AMDGPU offload plugin sets
// 'value' to the address of a field inside its HSA signal, so it is never null,
// while 'mailbox' is that signal's event mailbox pointer, which is null unless
// the signal is backed by an interrupt. Ringing such a doorbell must not
// dereference the null mailbox, and must still publish the pending work so a
// polling server makes progress.
TEST(LlvmLibcRPCDoorbell, NotifyWithoutMailbox) {
  ProcType Proc(port_count, buffer);

  pending = 0;
  Proc.doorbell->value = &pending;
  Proc.doorbell->mailbox = nullptr;
  Proc.doorbell->event_id = 0;

  Proc.notify(/*lane_mask=*/1);

  // notify() has no effect when built with MSVC.
#ifndef _MSC_VER
  EXPECT_EQ(pending, static_cast<uint64_t>(1));
#endif
}

// A doorbell that was never configured at all is ignored entirely.
TEST(LlvmLibcRPCDoorbell, NotifyWithoutDoorbell) {
  ProcType Proc(port_count, buffer);

  Proc.doorbell->value = nullptr;
  Proc.doorbell->mailbox = nullptr;
  Proc.doorbell->event_id = 0;

  Proc.notify(/*lane_mask=*/1);
}
