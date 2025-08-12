//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: This test is currently written in a way that is specific to libc++, but it's really trying to test a property
//       of the test framework, which isn't libc++ specific.
// REQUIRES: stdlib=libc++

// Make sure that the compile flags contain no module information.

// MODULE_DEPENDENCIES:

// RUN: echo "%{compile_flags}" | grep -v "std.pcm"
// RUN: echo "%{compile_flags}" | grep -v "std.compat.pcm"
