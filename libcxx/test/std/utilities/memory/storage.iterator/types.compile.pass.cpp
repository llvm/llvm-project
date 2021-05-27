//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_RAW_STORAGE_ITERATOR
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// raw_storage_iterator associated types

#include <memory>
#include <type_traits>

struct T;
typedef T* OutputIt;
typedef std::raw_storage_iterator<OutputIt, T> It;

static_assert(std::is_same<It::iterator_category, std::output_iterator_tag>::value, "");
static_assert(std::is_same<It::value_type, void>::value, "");
static_assert(std::is_same<It::difference_type, void>::value, "");
static_assert(std::is_same<It::pointer, void>::value, "");
static_assert(std::is_same<It::reference, void>::value, "");
