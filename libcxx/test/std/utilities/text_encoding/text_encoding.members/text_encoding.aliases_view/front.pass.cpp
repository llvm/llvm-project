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
// 1. An aliases_view from a single text_encoding object returns the same front()
// 2. An aliases_views of two text_encoding objects that represent the same ID but hold different names return the same front()
// 3. An aliases_views of two text_encoding objects that represent different IDs return different front() 

#include <cassert>
#include <cstdlib>
#include <text_encoding>

#include "platform_support.h" 
#include "test_macros.h"
#include "test_text_encoding.h"

using id = std::text_encoding::id;

int main() {

  {
    auto te = std::text_encoding(id::UTF8);
  
    auto view1 = te.aliases();
    auto view2 = te.aliases();
  
    assert(view1.front() == view2.front());
  }

  {
    auto te1 = std::text_encoding("ANSI_X3.4-1968");
    auto te2 = std::text_encoding("ANSI_X3.4-1986");

    auto view1 = te1.aliases(); 
    auto view2 = te2.aliases();

    assert(view1.front() == view2.front());
  }

  {
    
    auto te1 = std::text_encoding(id::UTF8);
    auto te2 = std::text_encoding(id::ASCII);
    
    auto view1 = te1.aliases();
    auto view2 = te2.aliases();

    assert(!(view1.front() == view2.front()));
  }

}
