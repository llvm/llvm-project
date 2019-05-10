// RUN: %clang_cc1 -E %s | FileCheck %s --match-full-lines --strict-whitespace
// CHECK:   zzap

// zzap is on a new line, should be indented.
#define BLAH  zzap
   BLAH

