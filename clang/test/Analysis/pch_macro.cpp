// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// DEFINE: %{emit-pch-action} = \
// DEFINE:   %clang_cc1 -x c++ -emit-pch -o %t/header.pch %t/header.h

// DEFINE: %{analyze-action} = \
// DEFINE:   %clang_analyze_cc1 -include-pch %t/header.pch \
// DEFINE:   -analyzer-checker=core,apiModeling,unix.StdCLibraryFunctions \
// DEFINE:   -verify %t/main.cpp

// RUN: %{emit-pch-action} -triple x86_64-apple-macosx10.15.0
// RUN: %{analyze-action}  -triple x86_64-apple-macosx10.15.0
// RUN: %{emit-pch-action}
// RUN: %{analyze-action}


//--- header.h


// Pre-compiled header

int foo();

// Literal data for macro values will be null as they are defined in a PCH
#define EOF -1
#define AT_FDCWD -2


//--- main.cpp


// Source file
// expected-no-diagnostics
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

