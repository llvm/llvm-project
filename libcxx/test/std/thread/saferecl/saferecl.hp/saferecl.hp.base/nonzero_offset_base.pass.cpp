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

// The hazard_pointer_obj_base<T, D> subobject need not sit at offset 0 of T: here a polymorphic base
// comes first. Protection, retirement and the deleter's argument must all still be expressed in terms of
// the T*, not of the base subobject, so a protected object survives reclamation passes and the deleter
// is eventually called with the same pointer retire() was called on.
//
// When an object is reclaimed is unspecified; like retire.pass.cpp this test only relies on the number
// of possibly-reclaimable objects being bounded, so retiring many objects makes *some* reclamation
// happen.

#include <hazard_pointer>
#include <atomic>
#include <cassert>
#include <cstdint>

#include "test_macros.h"

#if defined(TEST_IS_EXECUTED_IN_A_SLOW_ENVIRONMENT)
constexpr int N = 5000;
#else
constexpr int N = 100000; // comfortably above any sane reclamation bound
#endif

struct Prefix {
  virtual ~Prefix() = default; // makes Node polymorphic: the obj_base base cannot be at offset 0
  int tag           = 11;
};

struct Node;
struct NodeDeleter {
  void operator()(Node* p) const noexcept;
};

struct Node : Prefix, std::hazard_pointer_obj_base<Node, NodeDeleter> {
  Node* self_   = this;
  int value     = 42;
  bool watched_ = false; // the one object the test protects reports where it went
};

// Namespace scope: a deleter may run long after the block that retired the object has exited.
std::atomic<int> calls{0};
std::atomic<int> mismatches{0};
std::atomic<std::uintptr_t> watched_received{0};

void NodeDeleter::operator()(Node* p) const noexcept {
  ++calls;
  if (p->self_ != p || p->value != 42 || p->tag != 11)
    ++mismatches;
  if (p->watched_)
    watched_received.store(reinterpret_cast<std::uintptr_t>(p));
  delete p;
}

void retire_many() {
  for (int i = 0; i < N; ++i)
    (new Node)->retire();
}

int main(int, char**) {
  {
    // The premise of the test: the base subobject really is at a non-zero offset.
    Node probe;
    const void* derived = &probe;
    const void* base    = static_cast<std::hazard_pointer_obj_base<Node, NodeDeleter>*>(&probe);
    assert(derived != base);
  }

  Node* watched                     = new Node;
  watched->watched_                 = true;
  const std::uintptr_t watched_addr = reinterpret_cast<std::uintptr_t>(watched);
  {
    std::atomic<Node*> src{watched};
    std::hazard_pointer h = std::make_hazard_pointer();
    Node* p               = h.protect(src);
    assert(p == watched);
    src.store(nullptr);
    watched->retire();
    retire_many();
    assert(calls.load() > 0);             // reclamation did happen ...
    assert(watched_received.load() == 0); // ... but not of the protected object
    assert(watched->value == 42 && watched->tag == 11);
    h.reset_protection(); // epoch ends: watched becomes possibly-reclaimable
  }
  retire_many();
  // If it has been reclaimed by now -- unspecified, but the usual case -- the deleter got the T* that
  // retire() was called on, not the address of the base subobject.
  std::uintptr_t got = watched_received.load();
  assert(got == 0 || got == watched_addr);
  assert(mismatches.load() == 0);
  return 0;
}
