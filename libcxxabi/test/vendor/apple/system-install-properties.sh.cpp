//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: stdlib=apple-libc++

// This file checks various properties of the installation of libc++abi when built
// as a system library on Apple platforms.
//
// TODO: We should install to `<prefix>/usr` in CMake and check that path instead.

// Make sure we install the libc++abi headers in the right location.
// TODO: We don't currently install them, but we should.
//
// XRUNX: stat "%{install}/include/cxxabi.h"

// Make sure we install libc++abi.dylib in the right location.
//
// RUN: stat "%{install}/lib/libc++abi.dylib"

// Make sure we don't install a symlink from libc++abi.dylib to libc++abi.1.dylib,
// unlike what we do for libc++.dylib.
// TODO: We currently don't do that correctly in the CMake build.
//
// XRUNX: ! readlink "%{install}/lib/libc++abi.dylib"
// XRUNX: ! stat "%{install}/lib/libc++abi.1.dylib"

// Make sure the install_name is /usr/lib.
//
// In particular, make sure we don't use any @rpath in the load commands. When building as
// a system library, it is important to hardcode the installation paths in the dylib, because
// various tools like dyld and ld64 will treat us specially if they recognize us as being a
// system library.
//
// TODO: We currently don't do that correctly in the CMake build.
//
// XRUNX: otool -L "%{install}/lib/libc++abi.dylib" | grep '/usr/lib/libc++abi.dylib'
// XRUNX: ! otool -l "%{install}/lib/libc++abi.dylib" | grep -E "LC_RPATH|@loader_path|@rpath"

// Make sure the compatibility_version of libc++abi is 1.0.0. Failure to respect this can result
// in applications not being able to find libc++abi when they are loaded by dyld, if the
// compatibility version was bumped.
//
// RUN: otool -L "%{install}/lib/libc++abi.dylib" | grep "libc++abi.1.dylib" | grep "compatibility version 1.0.0"
