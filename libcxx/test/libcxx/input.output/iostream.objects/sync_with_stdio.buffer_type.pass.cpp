//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// static bool ios_base::sync_with_stdio(bool sync = true);

// This is a libc++ implementation detail: when sync_with_stdio(false) is used,
// libc++ swaps the buffers of cin/cout/clog (and the wide equivalents) for a
// basic_filebuf to avoid going through the C stdio layer. This test ensures
// that this optimization is applied by inspecting the type of the underlying
// buffer.

// The optimization is only applied when we have a filesystem, since fstream
// is not present otherwise. Also, the test itself requires RTTI.
// UNSUPPORTED: no-filesystem, no-rtti

// XFAIL: using-built-library-before-llvm-24

#include <cassert>
#include <fstream>
#include <iostream>

#include "test_macros.h"

int main(int, char**) {
  // By default the streams are synchronized, so they are not backed by a filebuf.
  assert(!dynamic_cast<std::filebuf*>(std::cin.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::cout.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::cerr.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::clog.rdbuf()));
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  assert(!dynamic_cast<std::wfilebuf*>(std::wcin.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wcout.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wcerr.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wclog.rdbuf()));
#endif

  // Turning synchronization off switches cin/cout/clog to a filebuf, but leaves
  // cerr untouched (it stays synchronized so it keeps flushing eagerly).
  std::ios_base::sync_with_stdio(false);
  assert(dynamic_cast<std::filebuf*>(std::cin.rdbuf()));
  assert(dynamic_cast<std::filebuf*>(std::cout.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::cerr.rdbuf()));
  assert(dynamic_cast<std::filebuf*>(std::clog.rdbuf()));
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  assert(dynamic_cast<std::wfilebuf*>(std::wcin.rdbuf()));
  assert(dynamic_cast<std::wfilebuf*>(std::wcout.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wcerr.rdbuf()));
  assert(dynamic_cast<std::wfilebuf*>(std::wclog.rdbuf()));
#endif

  // Turning it back on restores the original buffers.
  std::ios_base::sync_with_stdio(true);
  assert(!dynamic_cast<std::filebuf*>(std::cin.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::cout.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::cerr.rdbuf()));
  assert(!dynamic_cast<std::filebuf*>(std::clog.rdbuf()));
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  assert(!dynamic_cast<std::wfilebuf*>(std::wcin.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wcout.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wcerr.rdbuf()));
  assert(!dynamic_cast<std::wfilebuf*>(std::wclog.rdbuf()));
#endif

  return 0;
}
