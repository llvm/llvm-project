// RUN: %clang_cc1 -std=gnu99 %s -emit-llvm -o - | FileCheck %s
// CHECK: alloca {{.*}}, align 16

void adr(char *);

void vlaalign(int size)
{
    char __attribute__((aligned(16))) tmp[size+32];
    char tmp2[size+16];

    adr(tmp);
}
