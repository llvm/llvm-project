// Preamble detection test: see below for comments and test commands.
//* A BCPL comment that includes '/*'
#include <blah>
#ifndef FOO
#else
#ifdef BAR
#elif WIBBLE
#endif
#pragma unknown
#endif
#ifdef WIBBLE
#include "foo"
int bar;
#endif

// This test checks for detection of the preamble of a file, which
// includes all of the starting comments and #includes.

// RUN: %clang_cc1 -print-preamble %s > %t
// RUN: echo END. >> %t
// RUN: FileCheck < %t %s

// CHECK: // Preamble detection test: see below for comments and test commands.
// CHECK-NEXT: //* A BCPL comment that includes '/*'
// CHECK-NEXT: #include <blah>
// CHECK-NEXT: #ifndef FOO
// CHECK-NEXT: #else
// CHECK-NEXT: #ifdef BAR
// CHECK-NEXT: #elif WIBBLE
// CHECK-NEXT: #endif
// CHECK-NEXT: #pragma unknown
// CHECK-NEXT: #endif
// CHECK-NEXT: #ifdef WIBBLE
// CHECK-NEXT: #include "foo"
// CHECK-NEXT: END.
