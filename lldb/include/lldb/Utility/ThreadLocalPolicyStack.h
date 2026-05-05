//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_THREADLOCALPOLICYSTACK_H
#define LLDB_UTILITY_THREADLOCALPOLICYSTACK_H

#include <vector>

namespace lldb_private {

/// Generic per-thread policy stack.
///
/// The stack lives in thread_local storage. Each thread has its own stack,
/// initialized with a default-constructed base entry that is never popped.
/// RAII guards (Guard) push and pop policies.
///
/// For thread pool workers that don't inherit thread_local storage, the
/// policy must be passed into the lambda and pushed onto the worker
/// thread's stack when the task starts.
template <typename Policy> class ThreadLocalPolicyStack {
public:
  static ThreadLocalPolicyStack &GetForCurrentThread() {
    static thread_local ThreadLocalPolicyStack s_stack;
    return s_stack;
  }

  const Policy &Current() const { return m_stack.back(); }

  void Push(Policy policy) { m_stack.push_back(policy); }

  void Pop() {
    if (m_stack.size() > 1)
      m_stack.pop_back();
  }

  /// RAII guard that pushes a policy on construction and pops on destruction.
  class Guard {
  public:
    Guard(Policy policy) { GetForCurrentThread().Push(policy); }
    ~Guard() { GetForCurrentThread().Pop(); }

    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;
  };

private:
  std::vector<Policy> m_stack = {Policy{}};
};

} // namespace lldb_private

#endif // LLDB_UTILITY_THREADLOCALPOLICYSTACK_H
