// RUN: %clang_cc1 -E -verify %s

#pragma GCC dependency "./_Pragma-dependency.c"

#define self "./_Pragma-dependency.c"
// expected-error@+1 {{expected "FILENAME" or <FILENAME>}}
#pragma GCC dependency self

#define EMPTY
// FIXME: The diagnostic location is not correct, it's current in
//    #pragma GCC dependency EMPTY "./_Pragma-dependency.c"
//                                 ^ expected "FILENAME" or <FILENAME>
// The real location should be the start of EMPTY.
// expected-error@+1 {{expected "FILENAME" or <FILENAME>}}
#pragma GCC dependency EMPTY "./_Pragma-dependency.c"

#define DO_PRAGMA _Pragma 
#define STR "GCC dependency \"parse.y\"")
// expected-error@+1 {{'parse.y' file not found}}
  DO_PRAGMA (STR
