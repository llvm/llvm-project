// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %t/main.cpp
// RUN: not %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                        -analyzer-config 'unitybuild-file-extensions=; ;' \
// RUN:                        %t/main.cpp 2>&1 | FileCheck %s --check-prefix=EMPTY
// EMPTY: invalid input for analyzer-config option 'unitybuild-file-extensions', that expects a non-empty file extension
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-config 'unitybuild-file-extensions=.cpp;.impl;.c' \
// RUN:                    -verify=expected,all-extensions %t/main.cpp
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-config unitybuild-file-extensions=.impl \
// RUN:                    -verify=expected,impl-only %t/main.cpp

//--- foo.cpp
#include "nested.h"

int includedCppFunction(int x) {
  if (x == 0)
    return 1 / x; // all-extensions-warning{{Division by zero}}
  return 0;
}

// This bug is found through the call from main.cpp, even by the default run.
int calledIncludedCppFunction(int x) {
  return 1 / x; // expected-warning{{Division by zero}}
}

//--- nested.h
#include "nested.cpp"

//--- nested.cpp
// main.cpp includes this file through foo.cpp and nested.h.
int nestedCppFunction(int x) {
  if (x == 0)
    return 1 / x; // all-extensions-warning{{Division by zero}}
  return 0;
}

//--- header.h
int headerFunction(int x) {
  // .h extension is not included in any of the runs.
  if (x == 0)
    return 1 / x; // no-warning
  return 0;
}

// This bug is also found through a call from main.cpp, not as an entry point.
int calledHeaderFunction(int x) {
  return 1 / x; // expected-warning{{Division by zero}}
}

//--- file-extensions.impl
int includedImplFunction(int x) {
  if (x == 0)
    return 1 / x; // all-extensions-warning{{Division by zero}} impl-only-warning{{Division by zero}}
  return 0;
}

//--- file-extensions.c
int includedCFunction(int x) {
  if (x == 0)
    return 1 / x; // all-extensions-warning{{Division by zero}}
  return 0;
}

//--- main.cpp
// With the empty default, only the main file contributes entry points.
// The configured list also adds included files with matching extensions.
#include "foo.cpp"
#include "header.h"
#include "file-extensions.impl"
#include "file-extensions.c"

int mainFileFunction(int x) {
  if (x == 0)
    return 1 / x; // expected-warning{{Division by zero}}
  return 0;
}

int mainFileCaller(bool callHeader) {
  if (callHeader)
    return calledHeaderFunction(0);
  return calledIncludedCppFunction(0);
}
