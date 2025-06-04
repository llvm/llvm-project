// RUN: %clang_cc1 -triple x86_64-apple-macosx10.15.0 -emit-pch -o %t %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-macosx10.15.0 -include-pch %t \
// RUN:   -analyzer-checker=core,apiModeling,unix.StdCLibraryFunctions -verify %s
//
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_analyze_cc1 -include-pch %t \
// RUN:   -analyzer-checker=core,apiModeling,unix.StdCLibraryFunctions -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER
// Pre-compiled header

int foo();

// Literal data for macro values will be null as they are defined in a PCH
#define EOF -1
#define AT_FDCWD -2

#else
// Source file

int test() {
  // we need a function call here to initiate erroneous routine
  return foo(); // no-crash
}

// Test that StdLibraryFunctionsChecker can obtain the definition of
// AT_FDCWD even if it is from a PCH:
int faccessat(int, const char *, int, int);

void test_faccessat() {
  char fileSystemPath[10] = { 0 };

  if (0 != faccessat(AT_FDCWD, fileSystemPath, 2, 0x0030)) {}
}

#endif
