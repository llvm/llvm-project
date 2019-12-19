// Make sure the intrinsic headers compile cleanly with no warnings or errors.

// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -Wsystem-headers \
// RUN:   -Wcast-qual -fsyntax-only -flax-vector-conversions=none -x c++ -verify %s

// expected-no-diagnostics

#include <x86intrin.h>
