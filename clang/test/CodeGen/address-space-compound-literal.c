// RUN: %clang_cc1 -emit-llvm < %s | grep "internal addrspace(1) global i32 1"

typedef int a __attribute__((address_space(1)));
a* x = &(a){1};

