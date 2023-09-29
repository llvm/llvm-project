//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=libc++ && target={{.+}}-apple-{{.+}}

// This file checks various properties of the installation of libc++abi when built as
// part of the LLVM releases, on Apple platforms.

// Make sure we install the libc++abi headers in the right location.
//
// RUN: stat "%{include}/cxxabi.h"

// Make sure we install libc++abi.dylib in the right location.
//
// RUN: stat "%{lib}/libc++abi.1.dylib"

// Make sure we install a symlink from libc++abi.dylib to libc++abi.1.dylib.
//
// RUN: stat "%{lib}/libc++abi.dylib"
// RUN: readlink "%{lib}/libc++abi.dylib" | grep "libc++abi.1.dylib"

// Make sure we don't set a RPATH when we build the library. Since we are building a system
// library, we are supposed to find our dependencies at the usual system-provided locations,
// which doesn't require setting a RPATH in the library itself.
//
// RUN: otool -l "%{lib}/libc++abi.dylib" | grep --invert-match -e "LC_RPATH"

// Make sure the compatibility_version of libc++abi is 1.0.0. Failure to respect this can result
// in applications not being able to find libc++abi when they are loaded by dyld, if the
// compatibility version was bumped.
//
// RUN: otool -L "%{lib}/libc++abi.dylib" | grep "libc++abi.1.dylib" | grep "compatibility version 1.0.0"
