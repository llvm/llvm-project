//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "debug.h"

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

debug::~debug() = default;

debug::dummy_ostream::~dummy_ostream() = default;

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD
