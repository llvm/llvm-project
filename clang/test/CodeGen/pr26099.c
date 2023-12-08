// RUN: %clang_cc1 -ffreestanding %s -triple=i686-apple-darwin -target-feature +mmx -emit-llvm -o - -Wall -Werror
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +mmx -emit-llvm -o - -Wall -Werror
// REQUIRES: asserts

#include <x86intrin.h>

int __attribute__ ((__vector_size__ (8))) b;

void bar(int a)
{
  b = __builtin_ia32_vec_init_v2si (0, a);
}