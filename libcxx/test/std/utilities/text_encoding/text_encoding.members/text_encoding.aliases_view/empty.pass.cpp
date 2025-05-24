//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <text_encoding>

// REQUIRES: std-at-least-c++26

// UNSUPPORTED: no-localization

// class text_encoding

// text_encoding text_encoding::environment(); 

// Concerns:
// 1. An alias_view of a text_encoding object for "other" is empty
// 2. An alias_view of a text_encoding object for "unknown" is empty
// 3. An alias_view of a text_encoding object for a known encoding e.g. "UTF-8" is not empty

#include <cassert>
#include <cstdlib>
#include <ranges>
#include <text_encoding>

#include "platform_support.h" 
#include "test_macros.h"
#include "test_text_encoding.h"

using id = std::text_encoding::id;

int main(){

  {
    auto te = std::text_encoding(id::other);
    auto empty_range = te.aliases();
    
    assert(std::ranges::empty(empty_range));
    assert(empty_range.empty());
    assert(!bool(empty_range));
  }

  {
    auto te = std::text_encoding(id::unknown);
    auto empty_range = te.aliases();

    assert(std::ranges::empty(empty_range));
    assert(empty_range.empty());
    assert(!bool(empty_range));
  }

  {
    auto te = std::text_encoding(id::UTF8);
    auto range = te.aliases();

    assert(!std::ranges::empty(range));
    assert(!range.empty());
    assert(bool(range));
  }

}
