//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test greps for %t, which is expanded to a path with backslashes. When
// that is passed to grep, those backslashes would have to be escaped, which we
// don't do right now.
// UNSUPPORTED: windows

// Make sure that substitutions are performed inside additional compiler flags.

// ADDITIONAL_COMPILE_FLAGS: -I %t.1
// ADDITIONAL_COMPILE_FLAGS: -isystem %t.2 , -isysroot %t.3
// RUN: echo "%{compile_flags}" | grep -e '-I %t.1 -isystem %t.2 -isysroot %t.3'
