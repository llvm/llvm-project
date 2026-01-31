//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: darwin
// REQUIRES: stdlib=libc++ && !stdlib=apple-libc++

// This file checks various properties of the installation of libc++abi when built under
// a vanilla upstream configuration on Darwin platforms.

// Make sure we install the libc++abi headers in the right location.
//
// RUN: stat "%{include}/cxxabi.h"

// Make sure we install libc++abi.1.dylib in the right location.
//
// RUN: stat "%{lib}/libc++abi.1.dylib"

// Make sure we install a symlink from libc++abi.dylib to libc++abi.1.dylib.
//
// RUN: stat "%{lib-dir}/libc++abi.dylib"
// RUN: readlink "%{lib-dir}/libc++abi.dylib" | grep "libc++abi.1.dylib"
