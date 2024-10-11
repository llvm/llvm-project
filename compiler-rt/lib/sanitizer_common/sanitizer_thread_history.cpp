//===-- sanitizer_thread_history.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sanitizer_thread_history.h"

#include "sanitizer_stackdepot.h"
namespace __sanitizer {

void PrintThreadHistory(ThreadRegistry &registry, InternalScopedString &out) {
  ThreadRegistryLock l(&registry);
  // Stack traces are largest part of printout and they offten the same for
  // multiple threads, so we will deduplicate them.
  InternalMmapVector<const ThreadContextBase *> stacks;

  registry.RunCallbackForEachThreadLocked(
      [](ThreadContextBase *context, void *arg) {
        static_cast<decltype(&stacks)>(arg)->push_back(context);
      },
      &stacks);

  Sort(stacks.data(), stacks.size(),
       [](const ThreadContextBase *a, const ThreadContextBase *b) {
         if (a->stack_id < b->stack_id)
           return true;
         if (a->stack_id > b->stack_id)
           return false;
         return a->tid < b->tid;
       });

  auto describe_thread = [&](const ThreadContextBase *context) {
    if (!context) {
      out.Append("T-1");
      return;
    }
    out.AppendF("T%llu/%llu", context->unique_id, context->os_id);
    if (internal_strlen(context->name))
      out.AppendF(" (%s)", context->name);
  };

  auto get_parent =
      [&](const ThreadContextBase *context) -> const ThreadContextBase * {
    if (!context)
      return nullptr;
    ThreadContextBase *parent = registry.GetThreadLocked(context->parent_tid);
    if (!parent)
      return nullptr;
    if (parent->unique_id >= context->unique_id)
      return nullptr;
    return parent;
  };

  u32 stack_id = 0;
  for (const ThreadContextBase *context : stacks) {
    if (stack_id != context->stack_id) {
      StackDepotGet(stack_id).PrintTo(&out);
      stack_id = context->stack_id;
    }
    out.Append("Thread ");
    describe_thread(context);
    out.Append(" was created by ");
    describe_thread(get_parent(context));
    out.Append("\n");
  }
  if (!stacks.empty())
    StackDepotGet(stack_id).PrintTo(&out);
}

}  // namespace __sanitizer
