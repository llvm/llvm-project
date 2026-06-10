//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <variant>

namespace std {

const char* bad_variant_access::what() const noexcept { return "bad_variant_access"; }

const char* __bad_variant_access_with_msg::what() const noexcept { return __msg_; }

} // namespace std
