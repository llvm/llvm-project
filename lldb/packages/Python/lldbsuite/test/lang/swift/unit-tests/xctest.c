//===-- XCTest.c ------------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <stdio.h>

int main()
{
  void *test_case = dlopen("UnitTest.xctest/Contents/MacOS/test", RTLD_NOW);

  printf("%p\n", test_case); // Set breakpoint here

  return 0;
}
