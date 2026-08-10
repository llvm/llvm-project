//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// checks that CTAD works properly

#include <future>
#include <type_traits>

int func(char*);
static_assert(std::is_same_v<decltype(std::packaged_task{func}), std::packaged_task<int(char*)>>);

int funcn(char*) noexcept;
static_assert(std::is_same_v<decltype(std::packaged_task{funcn}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct Callable {
  int operator()(char*) noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{Callable<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{Callable<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableC {
  int operator()(char*) const noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableC<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableC<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableV {
  int operator()(char*) const noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableV<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableV<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableCV {
  int operator()(char*) const volatile noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableCV<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableCV<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableL {
  int operator()(char*) & noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableL<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableL<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableCL {
  int operator()(char*) const & noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableCL<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableCL<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableVL {
  int operator()(char*) const noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableVL<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableVL<false>{}}), std::packaged_task<int(char*)>>);

template <bool Noexcept>
struct CallableCVL {
  int operator()(char*) const volatile & noexcept(Noexcept);
};
static_assert(std::is_same_v<decltype(std::packaged_task{CallableCVL<true>{}}), std::packaged_task<int(char*)>>);
static_assert(std::is_same_v<decltype(std::packaged_task{CallableCVL<false>{}}), std::packaged_task<int(char*)>>);
