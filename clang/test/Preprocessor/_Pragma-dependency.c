// RUN: %clang_cc1 %s -E 2>&1 | grep 'DO_PRAGMA (STR'
// RUN: %clang_cc1 %s -E 2>&1 | grep '7:3'

#define DO_PRAGMA _Pragma 
#define STR "GCC dependency \"parse.y\"")
  // Test that this line is printed by caret diagnostics.
  DO_PRAGMA (STR
