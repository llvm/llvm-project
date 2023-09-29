//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=libc++ && target={{.+}}-apple-{{.+}}

// This file checks various properties of the installation of libc++ when built as
// part of the LLVM releases, on Apple platforms.

// Make sure we install the libc++ headers in the right location.
//
// RUN: stat "%{include}/__config"

// Make sure we install libc++.1.dylib and libc++experimental.a in the right location.
//
// RUN: stat "%{lib}/libc++.1.dylib"
// RUN: stat "%{lib}/libc++experimental.a"

// Make sure we install a symlink from libc++.dylib to libc++.1.dylib.
//
// RUN: stat "%{lib}/libc++.dylib"
// RUN: readlink "%{lib}/libc++.dylib" | grep "libc++.1.dylib"

// Make sure we don't set a RPATH when we build the library. Since we are building a system
// library, we are supposed to find our dependencies at the usual system-provided locations,
// which doesn't require setting a RPATH in the library itself.
//
// RUN: otool -l "%{lib}/libc++.1.dylib" | grep --invert-match -e "LC_RPATH"

// Make sure the compatibility_version of libc++ is 1.0.0.
// Failure to respect this can result in applications not being able to find libc++
// when they are loaded by dyld, if the compatibility version was bumped.
//
// RUN: otool -L "%{lib}/libc++.1.dylib" | grep "libc++.1.dylib" | grep "compatibility version 1.0.0"
