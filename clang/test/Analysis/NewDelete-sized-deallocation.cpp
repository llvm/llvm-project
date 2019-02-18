// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -fsized-deallocation
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -DINCLUDE_INCLUDES
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -DINCLUDE_INCLUDES -fsized-deallocation
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -DINCLUDE_INCLUDES -DTEST_INLINABLE_ALLOCATORS
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -DINCLUDE_INCLUDES -DTEST_INLINABLE_ALLOCATORS -fsized-deallocation

// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++14
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++14 -fsized-deallocation
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++14 -DINCLUDE_INCLUDES
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++14 -DINCLUDE_INCLUDES -fsized-deallocation
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++14 -DINCLUDE_INCLUDES -DTEST_INLINABLE_ALLOCATORS
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++14 -DINCLUDE_INCLUDES -DTEST_INLINABLE_ALLOCATORS -fsized-deallocation

// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++17
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++17 -fsized-deallocation
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++17 -DINCLUDE_INCLUDES
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++17 -DINCLUDE_INCLUDES -fsized-deallocation
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++17 -DINCLUDE_INCLUDES -DTEST_INLINABLE_ALLOCATORS
// RUN: %clang_analyze_cc1 -analyzer-checker=core,cplusplus -verify -analyzer-output=text %s -std=c++17 -DINCLUDE_INCLUDES -DTEST_INLINABLE_ALLOCATORS -fsized-deallocation

// Test all three: undeclared operator delete, operator delete forward-declared
// in the system header, operator delete defined in system header.
#ifdef INCLUDE_INCLUDES
// TEST_INLINABLE_ALLOCATORS is used within this include.
#include "Inputs/system-header-simulator-cxx.h"
#endif

void leak() {
  int *x = new int; // expected-note{{Memory is allocated}}
} // expected-warning{{Potential leak of memory pointed to by 'x'}}
  // expected-note@-1{{Potential leak of memory pointed to by 'x'}}

// This function was incorrectly diagnosed as leak under -fsized-deallocation
// because the sized operator delete was mistaken for a custom delete.
void no_leak() {
  int *x = new int; // no-note
  delete x;
} // no-warning
