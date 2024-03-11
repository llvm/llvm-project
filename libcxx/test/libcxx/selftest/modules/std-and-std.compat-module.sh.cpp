//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: clang-modules-build
// UNSUPPORTED: gcc

// XFAIL: has-no-cxx-module-support

// Make sure that the compile flags contain the expected elements.
// The tests only look for the expected components and not the exact flags.
// Otherwise changing the location of the module would break this test.

// MODULE_DEPENDENCIES: std std.compat

// RUN: echo "%{compile_flags}" | grep -- "-fmodule-file=std=.*/std.pcm .*/std.pcm"
// RUN: echo "%{compile_flags}" | grep -- "-fmodule-file=std.compat=.*/std.compat.pcm .*/std.compat.pcm"
