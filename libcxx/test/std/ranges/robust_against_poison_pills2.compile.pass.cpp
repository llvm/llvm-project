//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <compare>, <iterator>, <ranges>

// Ordinary unqualified lookup should not be performed.

namespace begin {}
namespace end {}
namespace cbegin {}
namespace cend {}
namespace rbegin {}
namespace rend {}
namespace crbegin {}
namespace crend {}
namespace size {}
namespace ssize {}
namespace reserve_hint {}
namespace empty {}
namespace data {}
namespace cdata {}
namespace iter_swap {}
namespace iter_move {}

#include <ranges>
