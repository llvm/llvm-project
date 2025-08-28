// XFAIL: target={{.*(riscv).*}}
// RUN: %clang -fenable-ripple -O2 %s -S -emit-llvm

#include <ripple.h>
extern int printf(const char*, ...);

void f() {
    ripple_block_t BS = ripple_set_block_shape(0, 32);
    size_t v0 = ripple_id(BS, 0);
    size_t reduce = ripple_reduceadd(0b1, v0);
    printf("%zu\n", reduce);
}
