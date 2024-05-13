// PR 1346
// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null
extern void bar(void *);

void f(void *cd) {
  bar(((void *)((unsigned long)(cd) ^ -1)));
}
