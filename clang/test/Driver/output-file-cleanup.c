// RUN: rm -f "%t.d" "%t1.s" "%t2.s" "%t3.s" "%t4.s" "%t5.s"

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

// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
// RUN: cd %t-dir

// RUN: touch %t-dir/1.c
// RUN: echo "invalid C code" > %t-dir/2.c
// RUN: not %clang -S %t-dir/1.c %t-dir/2.c
// RUN: test -f %t-dir/1.s
// RUN: test ! -f %t-dir/2.s

// RUN: touch %t-dir/1.c
// RUN: touch %t-dir/2.c
// RUN: chmod -r %t-dir/2.c
// RUN: not %clang -S %t-dir/1.c %t-dir/2.c
// RUN: test -f %t-dir/1.s
// RUN: test ! -f %t-dir/2.s
// RUN: rm -f %t-dir/2.c
