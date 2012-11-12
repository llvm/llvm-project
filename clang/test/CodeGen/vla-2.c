// RUN: %clang_cc1 -std=gnu99 %s -emit-llvm -o - | grep ".*alloca.*align 16"

extern void bar(int[]);

void foo(int a)
{
  int var[a] __attribute__((__aligned__(16)));
  bar(var);
  return;
}
