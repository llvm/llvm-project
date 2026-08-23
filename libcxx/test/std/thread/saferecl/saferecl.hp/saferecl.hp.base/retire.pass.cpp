//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

// <hazard_pointer>

// void retire(D d = D()) noexcept;
//   Move-assigns d to the deleter, then retires *this. When the object is reclaimed its deleter is
//   invoked with a pointer to the object. When that happens is unspecified; this test only relies on
//   the fact that the number of possibly-reclaimable objects is bounded, so retiring a large number of
//   objects makes *some* reclamation happen.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <memory>

#include "test_macros.h"

#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
constexpr int N = 5000;
#else
constexpr int N = 100000; // comfortably above any sane reclamation bound
#endif

// 1. The default deleter: default_delete<T>, i.e. `delete p`.
std::atomic<int> default_deleted{0};
struct Plain : std::hazard_pointer_obj_base<Plain> {
  ~Plain() { ++default_deleted; }
};

// 2. A stateful deleter, move-assigned by retire(); it checks it receives the object it was given.
struct Tally {
  std::atomic<int> calls{0};
  std::atomic<int> mismatches{0};
};
struct Recorded;
struct RecordingDeleter {
  Tally* tally = nullptr;
  void operator()(Recorded* p) const noexcept;
};
struct Recorded : std::hazard_pointer_obj_base<Recorded, RecordingDeleter> {
  Recorded* self = this;
  int payload    = 7;
};
void RecordingDeleter::operator()(Recorded* p) const noexcept {
  ++tally->calls;
  if (p->self != p || p->payload != 7)
    ++tally->mismatches;
  delete p;
}

// 3. retire(d) move-assigns d into the object's stored deleter: the deleter that eventually runs is the
//    move-assignment target, and it must have observed the assignment.
struct MoveTracked;
std::atomic<int> tracked_calls{0};
std::atomic<int> tracked_not_move_assigned{0};
struct MoveTrackingDeleter {
  bool assigned_by_move                      = false;
  MoveTrackingDeleter()                      = default;
  MoveTrackingDeleter(MoveTrackingDeleter&&) = default;
  MoveTrackingDeleter& operator=(MoveTrackingDeleter&&) noexcept {
    assigned_by_move = true;
    return *this;
  }
  void operator()(MoveTracked* p) const noexcept;
};
struct MoveTracked : std::hazard_pointer_obj_base<MoveTracked, MoveTrackingDeleter> {};
void MoveTrackingDeleter::operator()(MoveTracked* p) const noexcept {
  ++tracked_calls;
  if (!assigned_by_move)
    ++tracked_not_move_assigned;
  delete p;
}

// 4. A function pointer is a valid function object type for D.
struct FnObj;
using FnDeleter = void (*)(FnObj*);
struct FnObj : std::hazard_pointer_obj_base<FnObj, FnDeleter> {};
std::atomic<int> fn_calls{0};
void fn_delete(FnObj* p) {
  ++fn_calls;
  delete p;
}

int main(int, char**) {
  for (int i = 0; i < N; ++i)
    (new Plain)->retire();
  assert(default_deleted.load() > 0);

  Tally tally;
  for (int i = 0; i < N; ++i)
    (new Recorded)->retire(RecordingDeleter{&tally});
  assert(tally.calls.load() > 0);
  assert(tally.mismatches.load() == 0);

  for (int i = 0; i < N; ++i)
    (new MoveTracked)->retire(MoveTrackingDeleter{});
  assert(tracked_calls.load() > 0);
  assert(tracked_not_move_assigned.load() == 0);

  for (int i = 0; i < N; ++i)
    (new FnObj)->retire(&fn_delete);
  assert(fn_calls.load() > 0);

  return 0;
}
