//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=apple-libc++

// This file checks various properties of the installation of libc++ when built as
// a system library on Apple platforms.

// Make sure we install the libc++ headers in the right location.
//
// RUN: stat "%{include}/__config"

// Make sure we install libc++.1.dylib in the right location.
//
// RUN: stat "%{lib}/libc++.1.dylib"

// Make sure we install a symlink from libc++.dylib to libc++.1.dylib.
//
// RUN: stat "%{lib}/libc++.dylib"
// RUN: readlink "%{lib}/libc++.dylib" | grep "libc++.1.dylib"

// Make sure the install_name is /usr/lib.
//
// In particular, make sure we don't use any @rpath in the load commands. When building as
// a system library, it is important to hardcode the installation paths in the dylib, because
// various tools like dyld and ld64 will treat us specially if they recognize us as being a
// system library.
//
// TODO: We currently don't do that correctly in the CMake build.
//
// XRUNX: otool -L "%{lib}/libc++.1.dylib" | grep '/usr/lib/libc++.1.dylib'
// XRUNX: ! otool -l "%{lib}/libc++.1.dylib" | grep -E "LC_RPATH|@loader_path|@rpath"

// Make sure the compatibility_version of libc++ is 1.0.0.
// Failure to respect this can result in applications not being able to find libc++
// when they are loaded by dyld, if the compatibility version was bumped.
//
// RUN: otool -L "%{lib}/libc++.1.dylib" | grep "libc++.1.dylib" | grep "compatibility version 1.0.0"
