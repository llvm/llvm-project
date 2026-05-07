//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_POLICY_H
#define LLDB_TARGET_POLICY_H

#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace lldb_private {

class Stream;

/// Describes what view of the process a thread should see and what
/// operations it is allowed to perform.
///
/// This replaces ad-hoc checks like CurrentThreadIsPrivateStateThread() with
/// a unified, composable mechanism. Code consults the current policy on the
/// per-thread PolicyStack instead of comparing host thread identities.
///
/// One motivating case is frame providers, which layer a public illusion on
/// top of the private unwinder stack. The private state thread must see the
/// raw unwinder frames, while public clients see the augmented view. Rather
/// than checking thread identity at every callsite, the private state thread
/// pushes Policy::PrivateState() and the rest follows from the policy.
struct Policy {
  /// What view of the process this thread sees.
  enum class View {
    Public,  ///< Provider-augmented frames, public state, public run lock.
    Private, ///< Parent (unwinder) frames, private state, private run lock.
  };

  /// What operations this thread is allowed to perform.
  /// Enforced at specific callsites, not by the policy itself.
  struct Capabilities {
    bool can_evaluate_expressions = true;
    /// Whether expression evaluation may resume all threads to avoid
    /// deadlocks (e.g. when a lock is held by another thread).
    bool can_run_all_threads = true;
    /// Whether the expression runner may fall back to running all threads
    /// after a single-thread attempt times out.
    bool can_try_all_threads = true;
    bool can_run_breakpoint_actions = true;
    bool can_load_frame_providers = true;
    bool can_run_frame_recognizers = true;
  };

  View view = View::Public;
  Capabilities capabilities;

  static Policy PublicState() { return {}; }

  static Policy PrivateState() {
    Policy p;
    p.view = View::Private;
    p.capabilities.can_load_frame_providers = false;
    p.capabilities.can_run_frame_recognizers = false;
    return p;
  }

  static Policy PublicStateRunningExpression() {
    Policy p;
    p.capabilities.can_run_breakpoint_actions = false;
    return p;
  }

  void Dump(Stream &s) const;
};

/// Per-thread policy stack.
///
/// The stack lives in thread_local storage. Each thread has its own stack,
/// initialized with a default-constructed base entry that is never popped.
/// RAII guards (Guard) push and pop policies.
///
/// For thread pool workers that don't inherit thread_local storage, the
/// policy must be passed into the lambda and pushed onto the worker
/// thread's stack when the task starts.
class PolicyStack {
public:
  static PolicyStack &Get() {
    static thread_local PolicyStack s_stack;
    return s_stack;
  }

  Policy Current() const;

  void Push(Policy policy) { m_stack.push_back(std::move(policy)); }

  void Pop() {
    assert(!m_stack.empty() && "can't pop the base policy");
    m_stack.pop_back();
  }

  void Dump(Stream &s) const;

  /// RAII guard that pushes a policy on construction and pops on destruction.
  class Guard {
  public:
    explicit Guard(Policy policy) { Get().Push(std::move(policy)); }
    ~Guard() { Get().Pop(); }

    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;
  };

private:
  llvm::SmallVector<Policy> m_stack = {Policy{}};
};

} // namespace lldb_private

#endif // LLDB_TARGET_POLICY_H
