// Make sure the intrinsic headers compile cleanly with no warnings or errors.

// RUN: %clang_cc1 -ffreestanding -triple i386-unknown-unknown \
// RUN:    -Werror -Wsystem-headers -Wcast-qual \
// RUN:    -fsyntax-only -flax-vector-conversions=none -x c++ -verify %s

// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown \
// RUN:    -Werror -Wsystem-headers -Wcast-qual \
// RUN:    -fsyntax-only -flax-vector-conversions=none -x c++ -verify %s

// expected-no-diagnostics

#include <x86intrin.h>
