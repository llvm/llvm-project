//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: clang-modules-build
// UNSUPPORTED: apple-clang && c++03

// <cerrno>

// tests LWG 3869 deprecated macros.
//
// Note the macros may not be defined. When they are not defined the
// ifdef XXX does not trigger a deprecated message. So use them in the
// ifdef and test for 2 deprecated messages.

#include <cerrno>

#ifdef ENODATA
[[maybe_unused]] int nodata =
    ENODATA; // expected-warning@cerrno.syn.verify.cpp:* 2 {{macro 'ENODATA' has been marked as deprecated}}
#endif
#ifdef ENOSR
[[maybe_unused]] int nosr =
    ENOSR; // expected-warning@cerrno.syn.verify.cpp:* 2 {{macro 'ENOSR' has been marked as deprecated}}
#endif
#ifdef ENOSTR
[[maybe_unused]] int nostr =
    ENOSTR; // expected-warning@cerrno.syn.verify.cpp:* 2 {{macro 'ENOSTR' has been marked as deprecated}}
#endif
#ifdef ETIME
[[maybe_unused]] int timeout =
    ETIME; // expected-warning@cerrno.syn.verify.cpp:* 2 {{macro 'ETIME' has been marked as deprecated}}
#endif
