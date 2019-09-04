//===-- xctest.c ----------------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>

int main(int argc, const char **argv)
{
  char *dylib = strcat(dirname(argv[0]),"/UnitTest.xctest/Contents/MacOS/test");
  void *test_case = dlopen(dylib, RTLD_NOW);

  printf("%p\n", test_case); // Set breakpoint here

  return 0;
}
