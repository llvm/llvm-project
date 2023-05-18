// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -verify %s -ffreestanding
// expected-no-diagnostics

#include <stdint.h>
typedef unsigned long long uint64_t;
