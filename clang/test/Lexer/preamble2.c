// Preamble detection test: header with an include guard.
#ifndef HEADER_H
#define HEADER_H
#include "foo"
int bar;
#endif

// This test checks for detection of the preamble of a file, which
// includes all of the starting comments and #includes.

// RUN: %clang_cc1 -print-preamble %s > %t
// RUN: echo END. >> %t
// RUN: FileCheck < %t %s

// CHECK: // Preamble detection test: header with an include guard.
// CHECK-NEXT: #ifndef HEADER_H
// CHECK-NEXT: #define HEADER_H
// CHECK-NEXT: #include "foo"
// CHECK-NEXT: END.
