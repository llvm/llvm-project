// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s
// expected-no-diagnostics

typedef __UINTPTR_TYPE__ uintptr_t;

char *a = (void*)(uintptr_t)(void*)&a;
