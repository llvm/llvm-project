//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// check that <memory_resource> functions are marked [[nodiscard]]

#include <memory_resource>

#include "test_macros.h"

void test() {
  {
    std::pmr::memory_resource* r = std::pmr::null_memory_resource();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r->allocate(1);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r->is_equal(*r);

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::pmr::get_default_resource();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::pmr::new_delete_resource();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::pmr::null_memory_resource();
  }

  {
    std::pmr::monotonic_buffer_resource r;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r.upstream_resource();

    struct test_protected : public std::pmr::monotonic_buffer_resource {
      void test() {
        std::pmr::monotonic_buffer_resource other;

        // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
        do_allocate(94, 82);
        // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
        is_equal(other);
      }
    };
  }

  {
    std::pmr::polymorphic_allocator<int> a;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    a.allocate(1);
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    a.select_on_container_copy_construction();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    a.resource();
  }

  {
    std::pmr::synchronized_pool_resource r;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r.upstream_resource();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r.options();

    struct test_protected : public std::pmr::synchronized_pool_resource {
      void test() {
        std::pmr::synchronized_pool_resource other;

        // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
        do_allocate(94, 82);
        // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
        is_equal(other);
      }
    };
  }

  {
    std::pmr::unsynchronized_pool_resource r;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r.upstream_resource();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    r.options();

    struct test_protected : public std::pmr::unsynchronized_pool_resource {
      void test() {
        std::pmr::unsynchronized_pool_resource other;

        // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
        do_allocate(94, 82);
        // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
        is_equal(other);
      }
    };
  }
}
