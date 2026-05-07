//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <memory>
#include <cassert>
#include <type_traits>

struct Trivial {
  int id;
};

static_assert(noexcept(std::start_lifetime_as_array<int>(std::declval<void*>(), 5)));
static_assert(noexcept(std::start_lifetime_as_array<int>(std::declval<const volatile void*>(), 5)));

static_assert(std::is_same_v<decltype(std::start_lifetime_as_array<int>(std::declval<void*>(), 5)), int*>);
static_assert(std::is_same_v<decltype(std::start_lifetime_as_array<int>(std::declval<const void*>(), 5)), const int*>);
static_assert(
    std::is_same_v<decltype(std::start_lifetime_as_array<int>(std::declval<volatile void*>(), 5)), volatile int*>);
static_assert(std::is_same_v<decltype(std::start_lifetime_as_array<int>(std::declval<const volatile void*>(), 5)),
                             const volatile int*>);

int main(int, char**) {
  constexpr std::size_t count = 3;
  alignas(Trivial) unsigned char buffer[sizeof(Trivial) * count];

  void* p                  = buffer;
  const void* cp           = buffer;
  volatile void* vp        = buffer;
  const volatile void* cvp = buffer;

  Trivial* ptr                  = std::start_lifetime_as_array<Trivial>(p, count);
  const Trivial* cptr           = std::start_lifetime_as_array<Trivial>(cp, count);
  volatile Trivial* vptr        = std::start_lifetime_as_array<Trivial>(vp, count);
  const volatile Trivial* cvptr = std::start_lifetime_as_array<Trivial>(cvp, count);

  assert(static_cast<void*>(ptr) == buffer);
  assert(static_cast<const void*>(cptr) == buffer);
  assert(static_cast<volatile void*>(vptr) == buffer);
  assert(static_cast<const volatile void*>(cvptr) == buffer);

  return 0;
}
