//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// <shared_mutex>

// class shared_timed_mutex;

// shared_timed_mutex(const shared_timed_mutex&) = delete;

#include <shared_mutex>
#include <type_traits>

static_assert(!std::is_copy_constructible<std::shared_timed_mutex>::value, "");
