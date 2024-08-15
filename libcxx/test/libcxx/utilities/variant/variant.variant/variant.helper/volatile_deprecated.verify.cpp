//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <variant>

// UNSUPPORTED: c++03, c++11, c++17

#include <variant>

typedef std::variant<void, int> vars;

[[maybe_unused]] std::variant_alternative<0, vars> alt_test;

// expected-warning@*:* {{'__volatile_deprecated_since_cxx20_warning<volatile std::variant<void, int>>' is deprecated}}
[[maybe_unused]] std::variant_alternative<0, volatile vars> vol_alt_test;

// expected-warning@*:* {{'__volatile_deprecated_since_cxx20_warning<volatile std::variant<void, int>>' is deprecated}}
[[maybe_unused]] std::variant_alternative<0, volatile vars> const_vol_alt_test;

[[maybe_unused]] std::variant_size<vars> size_test;

// expected-warning@*:* {{'__volatile_deprecated_since_cxx20_warning<volatile std::variant<void, int>>' is deprecated}}
[[maybe_unused]] std::variant_size<volatile vars> vol_size_test;

// expected-warning@*:* {{'__volatile_deprecated_since_cxx20_warning<volatile std::variant<void, int>>' is deprecated}}
[[maybe_unused]] std::variant_size<volatile vars> const_vol_size_test;
