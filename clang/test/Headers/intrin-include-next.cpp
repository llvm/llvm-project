// Verify that, outside MSVC-compatible mode, <intrin.h> defers to another
// <intrin.h> later in the search path.

// A user-provided header in an include directory after clang's.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffreestanding -fsyntax-only \
// RUN:     -idirafter %S/Inputs/intrin-next -verify %s
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -ffreestanding -fsyntax-only \
// RUN:     -idirafter %S/Inputs/intrin-next -verify %s

// MinGW intrin.h.
// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -ffreestanding -fsyntax-only \
// RUN:     -idirafter %S/Inputs/intrin-next -verify %s

// expected-no-diagnostics
#include <intrin.h>
#ifndef INTRIN_NEXT_FOUND
#error "clang's <intrin.h> did not #include_next another intrin.h"
#endif
