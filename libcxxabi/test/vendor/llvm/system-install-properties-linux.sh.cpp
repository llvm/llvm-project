//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: linux
// REQUIRES: stdlib=libc++ && !stdlib=apple-libc++

// This file checks various properties of the installation of libc++abi when built under
// a vanilla upstream configuration on Linux platforms.

// Make sure we install the libc++abi headers in the right location.
//
// RUN: stat "%{include}/cxxabi.h"

// Make sure we install libc++abi.so.1.0 in the right location.
//
// RUN: stat "%{lib}/libc++abi.so.1.0"

// Make sure we install a symlink from libc++abi.so.1 to libc++abi.so.1.0.
//
// RUN: stat "%{lib}/libc++abi.so.1"
// RUN: readlink "%{lib}/libc++abi.so.1" | grep "libc++abi.so.1.0"

// Make sure we install a symlink from libc++abi.so to libc++abi.so.1.
//
// RUN: stat "%{lib}/libc++abi.so"
// RUN: readlink "%{lib}/libc++abi.so" | grep "libc++abi.so.1"
