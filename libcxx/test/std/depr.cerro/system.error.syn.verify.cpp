//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// These macros do not seem to behave as expected on all Apple platforms.
// Since the macros are not provided newer POSIX versions it is expected the
// macros will be retroactively removed from C++. (The deprecation was
// retroactively.)
// UNSUPPORTED: apple-clang && (c++03 || clang-modules-build)

// <system_error>

// enum errc {...}

// tests LWG 3869 deprecated enum members.

#include <system_error>

[[maybe_unused]] std::errc nodata =
    std::errc::no_message_available; // expected-warning {{'no_message_available' is deprecated}}
[[maybe_unused]] std::errc nosr =
    std::errc::no_stream_resources; // expected-warning {{'no_stream_resources' is deprecated}}
[[maybe_unused]] std::errc nostr   = std::errc::not_a_stream;   // expected-warning {{'not_a_stream' is deprecated}}
[[maybe_unused]] std::errc timeout = std::errc::stream_timeout; // expected-warning {{'stream_timeout' is deprecated}}
