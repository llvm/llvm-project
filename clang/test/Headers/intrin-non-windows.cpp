// Verify that <intrin.h> produces a clear diagnostic on non-Windows targets
// rather than silently failing via #include_next to a missing system header.
// See intrin-include-next.cpp for the case where there is a header to defer to.

// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffreestanding -fsyntax-only \
// RUN:     -Wsystem-headers -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -ffreestanding -fsyntax-only \
// RUN:     -Wsystem-headers -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -ffreestanding -fsyntax-only \
// RUN:     -Wsystem-headers -verify %s
// RUN: %clang_cc1 -triple x86_64-sie-ps5 -ffreestanding -fsyntax-only \
// RUN:     -Wsystem-headers -verify %s

// expected-error@intrin.h:* {{<intrin.h> is a Windows-only header and is not available on this platform}}
#include <intrin.h>
