// RUN: touch %t.o
// RUN: not %clang -DCRASH -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test ! -f %t.d

// RUN: touch %t.o
// RUN: not %clang -DMISSING -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test ! -f %t.d

// RUN: touch %t.o
// RUN: not %clang -o %t.o -MMD -MF %t.d %s
// RUN: test ! -f %t.o
// RUN: test -f %t.d

// REQUIRES: shell
// REQUIRES: crash-recovery

#ifdef CRASH
#pragma clang __debug crash
#elif defined(MISSING)
#include "nonexistent.h"
#else
invalid C code
#endif
