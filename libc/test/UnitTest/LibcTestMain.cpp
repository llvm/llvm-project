//===-- Main function for implementation of base class for libc unittests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibcTest.h"
#include "src/__support/CPP/string_view.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::testing::TestOptions;

namespace {

// A poor-man's getopt_long.
// Run unit tests with --gtest_color=no to disable printing colors, or
// --gtest_print_time to print timings in milliseconds only (as GTest does, so
// external tools such as Android's atest may expect that format to parse the
// output). Other command line flags starting with --gtest_ are ignored.
// Otherwise, the last command line arg is used as a test filter, if command
// line args are specified.
TestOptions parseOptions(int argc, char **argv) {
  TestOptions Options;

  for (int i = 1; i < argc; ++i) {
    string_view arg{argv[i]};

    if (arg == "--gtest_color=no")
      Options.PrintColor = false;
    else if (arg == "--gtest_print_time")
      Options.TimeInMs = true;
    // Ignore other unsupported gtest specific flags.
    else if (arg.starts_with("--gtest_"))
      continue;
    else
      Options.TestFilter = argv[i];
  }

  return Options;
}

} // anonymous namespace

// The C++ standard forbids declaring the main function with a linkage specifier
// outisde of 'freestanding' mode, only define the linkage for hermetic tests.
#if __STDC_HOSTED__
#define TEST_MAIN int main
#else
#define TEST_MAIN extern "C" int main
#endif

TEST_MAIN(int argc, char **argv, char **envp) {
  LIBC_NAMESPACE::testing::argc = argc;
  LIBC_NAMESPACE::testing::argv = argv;
  LIBC_NAMESPACE::testing::envp = envp;

  return LIBC_NAMESPACE::testing::Test::runTests(parseOptions(argc, argv));
}
