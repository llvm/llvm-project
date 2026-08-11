//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux
// REQUIRES: stdlib=libc++ && !stdlib=apple-libc++

// This file checks various properties of the installation of libc++ when built under
// a vanilla upstream configuration on Linux platforms.

// Make sure we install the libc++ headers in the right location.
//
// RUN: stat "%{include-dir}/__config"

// Make sure we install libc++.so.1.0 and libc++experimental.a in the right location.
//
// RUN: stat "%{lib-dir}/libc++.so.1.0"
// RUN: stat "%{lib-dir}/libc++experimental.a"

// Make sure we install a symlink from libc++.so.1 to libc++.so.1.0.
//
// RUN: stat "%{lib-dir}/libc++.so.1"
// RUN: readlink "%{lib-dir}/libc++.so.1" | grep "libc++.so.1.0"

// Make sure we install libc++.so in the right location. That may be a symlink or
// a linker script, so we don't check anything specific about that file.
//
// RUN: stat "%{lib-dir}/libc++.so"
