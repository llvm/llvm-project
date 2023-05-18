// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s
// expected-no-diagnostics

#include <stdint.h>

char *a = (void*)(uintptr_t)(void*)&a;
