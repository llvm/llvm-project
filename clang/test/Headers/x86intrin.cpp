// RUN: %clang_cc1 -fsyntax-only -ffreestanding %s -verify
// expected-no-diagnostics

#if defined(i386) || defined(__x86_64__)

// Include the metaheader that includes all x86 intrinsic headers.
extern "C++" {
#include <x86intrin.h>
}

#endif
