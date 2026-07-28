//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_POLICY_H
#define LLDB_UTILITY_POLICY_H

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <thread>

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
/// pushes Policy::CreatePrivateState() and the rest follows from the policy.
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

  /// Why a private-state policy is being pushed. Distinguishes a PST's
  /// ordinary private-state processing from a PST created to service
  /// RunThreadPlan expression evaluation, which must not run frame
  /// providers or recognizers (see StackFrameList::SelectMostRelevantFrame
  /// and Thread::GetStackFrameList).
  enum class PrivateStatePurpose {
    Default,
    RunningExpression,
  };

  View view = View::Public;
  Capabilities capabilities;

  /// @name Factories
  ///
  /// CreatePublicState is the baseline (returns default Policy{}). The
  /// transition factories below start from PolicyStack::Get().Current() and
  /// apply their named change on top.
  /// @{
  static Policy CreatePublicState();
  static Policy CreatePrivateState(
      PrivateStatePurpose purpose = PrivateStatePurpose::Default);
  static Policy CreatePublicStateRunningExpression();
  /// @}

  void Dump(Stream &s) const;
};

/// Per-thread policy stack.
///
/// The stack lives in thread_local storage. Each thread has its own stack,
/// initialized with a default-constructed base entry that is never popped.
/// RAII guards (Guard) push and pop policies.
///
/// Policies are pushed via named factory methods (PushPrivateState, etc.)
/// that return an RAII Guard. Direct Push is private to prevent callers
/// from assembling arbitrary capability combinations.
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

  void Dump(Stream &s) const;

  /// RAII guard that pops a policy on destruction.
  ///
  /// A Guard is bound to the thread that created it: the policy stack lives
  /// in thread_local storage, so popping from a different thread would
  /// corrupt that thread's stack. Guards may be moved, but only on the
  /// owning thread; a cross-thread move or destruction is a fatal error.
  class Guard {
    friend class PolicyStack;

  public:
    ~Guard();
    Guard(Guard &&other);
    Guard &operator=(Guard &&other);

    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;

  private:
    Guard() : m_thread_id(std::this_thread::get_id()), m_active(true) {}
    std::thread::id m_thread_id;
    bool m_active = false;
  };

  /// All Push* methods delegate to the named static factories on Policy,
  /// which already inherit from Current(). So the pushed policy preserves
  /// existing stack state instead of resetting unrelated fields.

  [[nodiscard]] Guard
  PushPrivateState(Policy::PrivateStatePurpose purpose =
                       Policy::PrivateStatePurpose::Default) {
    Push(Policy::CreatePrivateState(purpose));
    return Guard();
  }

  [[nodiscard]] Guard PushPublicStateRunningExpression() {
    Push(Policy::CreatePublicStateRunningExpression());
    return Guard();
  }

private:
  void Push(Policy policy) { m_stack.push_back(std::move(policy)); }

  void Pop() {
    assert(!m_stack.empty() && "can't pop the base policy");
    m_stack.pop_back();
  }

  llvm::SmallVector<Policy> m_stack = {Policy{}};
};

} // namespace lldb_private

#endif // LLDB_UTILITY_POLICY_H
