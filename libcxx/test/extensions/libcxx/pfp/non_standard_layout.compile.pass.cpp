//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that enabling PFP causes certain vocabulary types to become non-standard
// layout (i.e. PFP enabled on their fields).
// REQUIRES: pfp

#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

static_assert(!std::is_standard_layout<std::map<int, int>>::value);
static_assert(!std::is_standard_layout<std::set<int>>::value);
static_assert(!std::is_standard_layout<std::vector<int>>::value);
static_assert(!std::is_standard_layout<std::string>::value);
static_assert(!std::is_standard_layout<std::unique_ptr<int>>::value);
static_assert(!std::is_standard_layout<std::unique_ptr<int[]>>::value);
static_assert(!std::is_standard_layout<std::shared_ptr<int>>::value);
static_assert(!std::is_standard_layout<std::weak_ptr<int>>::value);
