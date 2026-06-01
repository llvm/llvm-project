//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// Check that we're consistently using the same allocation functions to
// allocate/deallocate/construct/destroy objects in std::any.
// See https://llvm.org/PR45099 for details.

#include <any>
#include <cassert>
#include <cstddef>
#include <new>

// Make sure we don't fit in std::any's SBO
int allocated_count   = 0;
int constructed_count = 0;

struct Large {
  Large() { ++constructed_count; }

  Large(const Large&) { ++constructed_count; }

  ~Large() { --constructed_count; }

  char big[sizeof(std::any) + 1];

  static void* operator new(size_t n) {
    ++allocated_count;
    return ::operator new(n);
  }

  static void operator delete(void* ptr) {
    --allocated_count;
    ::operator delete(ptr);
  }
};

// Make sure we fit in std::any's SBO
struct Small {
  Small() { ++constructed_count; }

  Small(const Small&) { ++constructed_count; }

  ~Small() { --constructed_count; }

  static void* operator new(size_t n) {
    ++allocated_count;
    return ::operator new(n);
  }

  static void operator delete(void* ptr) {
    --allocated_count;
    ::operator delete(ptr);
  }
};

int main(int, char**) {
  // Test large types
  {
    [[maybe_unused]] std::any a = Large();
    assert(constructed_count == 1);
  }
  assert(allocated_count == 0);
  assert(constructed_count == 0);

  // Test small types
  {
    [[maybe_unused]] std::any a = Small();
    assert(constructed_count == 1);
  }
  assert(allocated_count == 0);
  assert(constructed_count == 0);

  return 0;
}
