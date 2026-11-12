//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, no-exceptions

// Verify that std::bad_variant_access::what() returns a message describing the
// cause of the failure for std::get and std::visit.

#include <cassert>
#include <cstring>
#include <variant>

struct ThrowOnMove {
  ThrowOnMove() = default;
  ThrowOnMove(ThrowOnMove&&) { throw 0; }
};

int main(int, char**) {
  // std::get<I> on the wrong alternative
  {
    std::variant<int, long> v(42);
    try {
      (void)std::get<1>(v);
      assert(false);
    } catch (const std::bad_variant_access& e) {
      assert(std::strcmp(e.what(), "std::get: wrong alternative for variant") == 0);
    }
  }
  // std::get<T> on the wrong alternative
  {
    std::variant<int, long> v(42);
    try {
      (void)std::get<long>(v);
      assert(false);
    } catch (const std::bad_variant_access& e) {
      assert(std::strcmp(e.what(), "std::get: wrong alternative for variant") == 0);
    }
  }
  // std::get<I> on a valueless variant
  {
    std::variant<int, ThrowOnMove> v(42);
    try {
      v.emplace<1>(ThrowOnMove{});
    } catch (...) {
    }
    assert(v.valueless_by_exception());
    try {
      (void)std::get<0>(v);
      assert(false);
    } catch (const std::bad_variant_access& e) {
      assert(std::strcmp(e.what(), "std::get: variant is valueless") == 0);
    }
  }
  // std::get<T> on a valueless variant
  {
    std::variant<int, ThrowOnMove> v(42);
    try {
      v.emplace<1>(ThrowOnMove{});
    } catch (...) {
    }
    assert(v.valueless_by_exception());
    try {
      (void)std::get<int>(v);
      assert(false);
    } catch (const std::bad_variant_access& e) {
      assert(std::strcmp(e.what(), "std::get: variant is valueless") == 0);
    }
  }
  // std::visit on a valueless variant
  {
    std::variant<int, ThrowOnMove> v(42);
    try {
      v.emplace<1>(ThrowOnMove{});
    } catch (...) {
    }
    assert(v.valueless_by_exception());
    try {
      std::visit([](auto&&) {}, v);
      assert(false);
    } catch (const std::bad_variant_access& e) {
      assert(std::strcmp(e.what(), "std::visit: variant is valueless") == 0);
    }
  }
  return 0;
}
