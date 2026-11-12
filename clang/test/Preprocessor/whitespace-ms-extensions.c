// RUN: %clang_cc1 -E -P %s -o - | FileCheck %s
// RUN: %clang_cc1 -E -P -fms-extensions %s -o - | FileCheck %s --check-prefix=MSEXT

// -fms-extensions changes __pragma into #pragma
// Ensure that there is a newline after the #pragma line.

#define MACRO        \
    text             \
    __pragma(PRAGMA) \
    after

before MACRO text


// CHECK:      before text __pragma(PRAGMA) after text

// MSEXT:      before text
// MSEXT-NEXT: #pragma PRAGMA
// MSEXT-NEXT: after text
