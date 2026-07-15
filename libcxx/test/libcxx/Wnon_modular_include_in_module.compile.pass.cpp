//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.*}}-apple-{{.*}}
// UNSUPPORTED: c++03

// This test ensures that libc++ supports being compiled with modules enabled and with
// -Wnon-modular-include-in-module. This effectively checks that we don't include any
// non-modular header from the library.
//
// Since most underlying platforms are not modularized properly, this test currently only
// works on Apple platforms.

// ADDITIONAL_COMPILE_FLAGS: -Wnon-modular-include-in-module -Wsystem-headers-in-module=std -fmodules -fcxx-modules

#include <vector>
