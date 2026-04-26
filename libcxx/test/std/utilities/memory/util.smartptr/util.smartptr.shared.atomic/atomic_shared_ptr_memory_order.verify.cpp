//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: diagnose-if-support
// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

#include <atomic>
#include <memory>

void test() {
  std::atomic<std::shared_ptr<int>> atom;
  std::shared_ptr<int> expected;
  std::shared_ptr<int> desired;

  atom.store(desired,
             std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
  atom.store(desired,
             std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
  atom.store(desired,
             std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}

  (void)atom.load(
      std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
  (void)atom.load(
      std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}

  atom.wait(expected,
            std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
  atom.wait(expected,
            std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}

  atom.compare_exchange_strong(
      expected,
      desired,
      std::memory_order_seq_cst,
      std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
  atom.compare_exchange_strong(
      expected,
      desired,
      std::memory_order_seq_cst,
      std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
  atom.compare_exchange_weak(
      expected,
      desired,
      std::memory_order_seq_cst,
      std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
  atom.compare_exchange_weak(
      expected,
      desired,
      std::memory_order_seq_cst,
      std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
}
