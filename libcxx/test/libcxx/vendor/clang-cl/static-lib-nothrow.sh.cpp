//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-windows-msvc

// The VCRuntime library owns std::nothrow when libc++ uses its ABI. Defining
// the object in libc++.lib as well makes whole-archive links fail with a
// duplicate symbol.

// RUN: llvm-nm --defined-only "%{lib-dir}/libc++.lib" | not grep -F '?nothrow@std@@3Unothrow_t@1@B'
// RUN: llvm-nm --defined-only "%{lib-dir}/libc++.lib" | grep -F '?__throw_bad_alloc@std@@YAXXZ'
