//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: no-threads

#include <mutex>

struct Empty {};

static_assert(sizeof(std::scoped_lock<Empty>) == sizeof(void*));
static_assert(sizeof(std::scoped_lock<Empty, Empty>) == 2 * sizeof(void*));

static_assert(sizeof(std::scoped_lock<std::mutex>) == sizeof(void*));
static_assert(sizeof(std::scoped_lock<std::mutex, std::mutex>) == 2 * sizeof(void*));
static_assert(sizeof(std::scoped_lock<std::mutex, Empty>) == 2 * sizeof(void*));

struct Big {
  char arr[10000];
};

static_assert(sizeof(std::scoped_lock<Big>) == sizeof(void*));
static_assert(sizeof(std::scoped_lock<Big, Big>) == 2 * sizeof(void*));
static_assert(sizeof(std::scoped_lock<Big, Big, Big>) == 3 * sizeof(void*));
