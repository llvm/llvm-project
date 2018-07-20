// Make sure the intrinsic headers compile cleanly with no warnings or errors.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wsystem-headers \
// RUN:   -fsyntax-only -x c++ -Wno-ignored-attributes -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wsystem-headers \
// RUN:   -fsyntax-only -x c++ -Wno-ignored-attributes -target-feature +f16c \
// RUN:   -verify %s

// expected-no-diagnostics

// Dont' include mm_malloc.h. It's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>
