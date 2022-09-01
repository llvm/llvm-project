// RUN: %clang -emit-llvm -fyk-noinline-funcs-with-loops -c -O0 -o - %s | llvm-dis | grep "call .*@never_aot_inline("
// RUN: %clang -emit-llvm -fyk-noinline-funcs-with-loops -c -O1 -o - %s | llvm-dis | grep "call .*@never_aot_inline("
// RUN: %clang -emit-llvm -fyk-noinline-funcs-with-loops -c -O2 -o - %s | llvm-dis | grep "call .*@never_aot_inline("
// RUN: %clang -emit-llvm -fyk-noinline-funcs-with-loops -c -O3 -o - %s | llvm-dis | grep "call .*@never_aot_inline("

#include <stdio.h>

static void never_aot_inline(int i) {
  while (i--)
    putchar('.');
}

int main(int argc, char **argv) {
  never_aot_inline(argc);
}
