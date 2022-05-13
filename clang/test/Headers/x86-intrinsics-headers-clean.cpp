// Make sure the intrinsic headers compile cleanly with no warnings or errors.

// RUN: %clang_cc1 -ffreestanding -triple i386-unknown-unknown \
// RUN:    -Wextra -Werror -Wsystem-headers -Wsign-conversion -Wcast-qual -Wdocumentation -Wdocumentation-pedantic -Wdocumentation-unknown-command \
// RUN:    -fsyntax-only -fretain-comments-from-system-headers -flax-vector-conversions=none -x c++ -verify %s

// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown \
// RUN:    -Wextra -Werror -Wsystem-headers -Wsign-conversion -Wcast-qual -Wdocumentation -Wdocumentation-pedantic -Wdocumentation-unknown-command \
// RUN:    -fsyntax-only -fretain-comments-from-system-headers -flax-vector-conversions=none -x c++ -verify %s

// expected-no-diagnostics

#include <x86intrin.h>
