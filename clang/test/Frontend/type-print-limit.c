// RUN: %clang_cc1 -ftype-print-limit=4000000000 -verify -fsyntax-only %s
// RUN: not %clang_cc1 -ftype-print-limit=-1 -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=BAD
// RUN: not %clang_cc1 -ftype-print-limit=abc -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=BAD-ABC

// expected-no-diagnostics

// BAD: error: invalid integral value '-1' in '-ftype-print-limit=-1'
// BAD-ABC: error: invalid integral value 'abc' in '-ftype-print-limit=abc'
