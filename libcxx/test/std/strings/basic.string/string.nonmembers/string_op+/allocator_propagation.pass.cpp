//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This test ensures that we properly propagate allocators, per https://wg21.link/p1165r1

#include <cassert>
#include <string>

#include "test_macros.h"

template <class T>
class soccc_allocator {
  int* soccc_count;
  int self_soccc_count;

public:
  using value_type = T;

  constexpr explicit soccc_allocator(int* soccc_count_, int self_coccc_count_ = 0)
      : soccc_count(soccc_count_), self_soccc_count(self_coccc_count_) {}

  template <class U>
  constexpr soccc_allocator(const soccc_allocator<U>& a) : soccc_count(a.soccc_count) {}

  constexpr T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  constexpr void deallocate(T* p, std::size_t s) { std::allocator<T>().deallocate(p, s); }

  constexpr soccc_allocator select_on_container_copy_construction() const {
    *soccc_count += 1;
    return soccc_allocator(soccc_count, self_soccc_count + 1);
  }

  constexpr auto get_soccc() { return soccc_count; }
  constexpr auto get_self_soccc() { return self_soccc_count; }

  typedef std::true_type propagate_on_container_copy_assignment;
  typedef std::true_type propagate_on_container_move_assignment;
  typedef std::true_type propagate_on_container_swap;
};

template <class CharT>
bool test() {
  using S = std::basic_string<CharT, std::char_traits<CharT>, soccc_allocator<CharT>>;
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs + rhs;
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 1);
    assert(soccc_lhs == 1);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs + std::move(rhs);
    assert(r.get_allocator().get_soccc() == &soccc_rhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = std::move(lhs) + rhs;
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = std::move(lhs) + std::move(rhs);
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs + rhs.data();
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 1);
    assert(soccc_lhs == 1);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs + rhs[0];
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 1);
    assert(soccc_lhs == 1);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = std::move(lhs) + rhs.data();
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = std::move(lhs) + rhs[0];
    assert(r.get_allocator().get_soccc() == &soccc_lhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs.data() + rhs;
    assert(r.get_allocator().get_soccc() == &soccc_rhs);
    assert(r.get_allocator().get_self_soccc() == 1);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 1);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs[0] + rhs;
    assert(r.get_allocator().get_soccc() == &soccc_rhs);
    assert(r.get_allocator().get_self_soccc() == 1);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 1);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs.data() + std::move(rhs);
    assert(r.get_allocator().get_soccc() == &soccc_rhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }
  {
    int soccc_lhs = 0;
    int soccc_rhs = 0;
    S lhs(soccc_allocator<CharT>{&soccc_lhs});
    S rhs(soccc_allocator<CharT>{&soccc_rhs});
    auto r = lhs[0] + std::move(rhs);
    assert(r.get_allocator().get_soccc() == &soccc_rhs);
    assert(r.get_allocator().get_self_soccc() == 0);
    assert(soccc_lhs == 0);
    assert(soccc_rhs == 0);
  }

  return true;
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
#if TEST_STD_VER > 17
  // static_assert(test<char>());
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  // static_assert(test<wchar_t>());
#endif
#endif

  return 0;
}
