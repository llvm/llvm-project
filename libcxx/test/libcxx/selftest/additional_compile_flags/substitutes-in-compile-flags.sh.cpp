//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test greps for %t, which is expanded to a path with backslashes. When
// that is passed to grep, those backslashes must be escaped. We escape those
// within the pattern into a file and use this file with 'grep'.

// Make sure that substitutions are performed inside additional compiler flags.

// ADDITIONAL_COMPILE_FLAGS: -I %t.1
// ADDITIONAL_COMPILE_FLAGS: -isystem %t.2 , -isysroot %t.3
// RUN: echo "-I %t.1 -isystem %t.2 -isysroot %t.3" | sed "s/\\\/\\\\\\\/g" > %t.escaped.grep
// RUN: echo "%{compile_flags}" | grep -e -f %t.escaped.grep
