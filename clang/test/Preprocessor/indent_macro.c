// RUN: %clang_cc1 -E %s | grep '^   zzap$'

// zzap is on a new line, should be indented.
#define BLAH  zzap
   BLAH

