// RUN: %clang_cc1 -emit-llvm %s -o - | grep volatile
// PR1647

void foo(volatile int *p)
{
p[0] = 0;
}
