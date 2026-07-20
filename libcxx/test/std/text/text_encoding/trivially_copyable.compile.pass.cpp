//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <text_encoding>
// Class text_encoding is a trivially copyable type ([basic.types.general]).

#include <text_encoding>
#include <type_traits>

static_assert(std::is_trivially_copyable_v<std::text_encoding>);
