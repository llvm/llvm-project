// RUN: %clang_cc1 -triple i386-apple-darwin10 -target-cpu yonah -fsyntax-only -verify -std=c89 %s
// expected-no-diagnostics

// FIXME: Disable inclusion of mm_malloc.h, our current implementation is broken
// on win32 since we don't generally know how to find errno.h.

#define __MM_MALLOC_H

// PR6658
#include <xmmintrin.h>

