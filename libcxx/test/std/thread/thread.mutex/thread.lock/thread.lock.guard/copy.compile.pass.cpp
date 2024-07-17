//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class lock_guard;

// lock_guard(lock_guard const&) = delete;

#include <mutex>

#include "types.h"

static_assert(!std::is_copy_constructible<std::lock_guard<MyMutex> >::value, "");
