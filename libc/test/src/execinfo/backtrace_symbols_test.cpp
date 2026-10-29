//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for backtrace_symbols
///
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "src/execinfo/backtrace_symbols.h"
#include "src/string/strlen.h"
#include "src/string/strstr.h"
#include "test/UnitTest/Test.h"

namespace {

// Somewhere to point at. What it does is of no interest; only that it has an
// address inside the program.
[[gnu::noinline]] int named_function() { return 7; }

} // anonymous namespace

// Every address is named, whether or not anything is known about it, and the
// names are laid out in one block: the array of pointers first and the text
// after it. The block comes from the allocator, which on Linux is scudo and
// is not linked into a test, so it comes from the one the test framework
// provides and is never handed back.
TEST(LlvmLibcBacktraceSymbolsTest, NamesEachAddress) {
  void *buffer[2] = {reinterpret_cast<void *>(&named_function),
                     reinterpret_cast<void *>(&buffer)};

  char **names = LIBC_NAMESPACE::backtrace_symbols(buffer, 2);
  ASSERT_FALSE(names == nullptr);
  for (int i = 0; i < 2; ++i) {
    ASSERT_FALSE(names[i] == nullptr);
    ASSERT_GT(LIBC_NAMESPACE::strlen(names[i]), size_t(0));
    // The address in brackets is there even where nothing else is known.
    ASSERT_FALSE(LIBC_NAMESPACE::strstr(names[i], "[0x") == nullptr);
  }

  // The text follows the array of pointers in the same block.
  ASSERT_GT(reinterpret_cast<char *>(names[0]),
            reinterpret_cast<char *>(names));
}

TEST(LlvmLibcBacktraceSymbolsTest, NothingToName) {
  void *buffer[1] = {};
  ASSERT_TRUE(LIBC_NAMESPACE::backtrace_symbols(buffer, 0) == nullptr);
  ASSERT_TRUE(LIBC_NAMESPACE::backtrace_symbols(nullptr, 4) == nullptr);
}
