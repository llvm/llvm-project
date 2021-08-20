// RUN: %clang_cc1 %s -verify
// RUN: not %clang_cc1 %s 2>&1 | FileCheck %s

// There is a special characters on the following line, which is used as a
// marker character for diagnostic printing.  Ensure diagnostics involving
// this character does not cause problems with the diagnostic printer.
#error Hi  Bye
// expected-error@-1 {{Hi <U+007F> Bye}}

// CHECK: error: Hi <U+007F> Bye
// CHECK: #error Hi <U+007F> Bye
