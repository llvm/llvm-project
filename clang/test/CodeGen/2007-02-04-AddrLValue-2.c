// RUN: %clang_cc1 %s -O3 -emit-llvm -o -
// PR1173

struct S { char s; };
struct T { struct S t; };

struct S *const p = &((struct T * const) (0x4000))->t;

void
foo (void)
{
  p->s = 0;
}
