//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// class recursive_mutex;

// recursive_mutex& operator=(const recursive_mutex&) = delete;

#include <mutex>
#include <type_traits>

static_assert(!std::is_copy_assignable<std::recursive_mutex>::value, "");
