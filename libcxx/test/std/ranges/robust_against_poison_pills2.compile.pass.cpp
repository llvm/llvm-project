//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

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

#include <ranges>
