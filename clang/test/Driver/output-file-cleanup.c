// RUN: rm -f "%t.d" "%t1.s" "%t2.s" "%t3.s" "%t4.s" "%t5.s"
//
// RUN: touch %t.s
// RUN: not %clang -S -DCRASH -o %t.s -MMD -MF %t.d %s
// RUN: test ! -f %t.s
// RUN: test ! -f %t.d

// RUN: touch %t.s
// RUN: not %clang -S -DMISSING -o %t.s -MMD -MF %t.d %s
// RUN: test ! -f %t.s
// RUN: test ! -f %t.d

// RUN: touch %t.s
// RUN: not %clang -S -o %t.s -MMD -MF %t.d %s
// RUN: test ! -f %t.s
// RUN: test -f %t.d

// REQUIRES: crash-recovery

#ifdef CRASH
#pragma clang __debug crash
#elif defined(MISSING)
#include "nonexistent.h"
#else
invalid C code
#endif

// RUN: touch %t1.c
// RUN: echo "invalid C code" > %t2.c
// RUN: cd %T && not %clang -S %t1.c %t2.c
// RUN: test -f %t1.s
// RUN: test ! -f %t2.s

// RUN: touch %t1.c
// RUN: touch %t2.c
// RUN: chmod -r %t2.c
// RUN: cd %T && not %clang -S %t1.c %t2.c
// RUN: test -f %t1.s
// RUN: test ! -f %t2.s

// When given multiple .c files to compile, clang compiles them in order until
// it hits an error, at which point it stops.
//
// RUN: touch %t1.c
// RUN: echo "invalid C code" > %t2.c
// RUN: touch %t3.c
// RUN: echo "invalid C code" > %t4.c
// RUN: touch %t5.c
// RUN: cd %T && not %clang -S %t1.c %t2.c %t3.c %t4.c %t5.c
// RUN: test -f %t1.s
// RUN: test ! -f %t2.s
// RUN: test ! -f %t3.s
// RUN: test ! -f %t4.s
// RUN: test ! -f %t5.s
