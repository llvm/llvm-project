// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s
// PR 3518
// Some of the objects were coming out as uninitialized (external) before 3518
// was fixed.  Internal names are different between llvm-gcc and clang so they
// are not tested.

extern void abort (void);

// CHECK: @.compoundliteral = internal global %struct.A { i32 1, i32 2 }
// CHECK: @.compoundliteral.1 = internal global %struct.A { i32 3, i32 4 }
// CHECK: @.compoundliteral.2 = internal global %struct.B { %struct.A* @.compoundliteral, %struct.A* @.compoundliteral.1 }
// CHECK: @.compoundliteral.3 = internal global %struct.A { i32 5, i32 6 }

struct A { int i; int j; };
struct B { struct A *a; struct A *b; };
struct C { struct B *c; struct A *d; };
struct C e = { &(struct B) { &(struct A) { 1, 2 }, &(struct A) { 3, 4 } }, &(struct A) { 5, 6 } };

int
main (void)
{
  if (e.c->a->i != 1 || e.c->a->j != 2)
    abort ();
  if (e.c->b->i != 3 || e.c->b->j != 4)
    abort ();
  if (e.d->i != 5 || e.d->j != 6)
    abort ();
  return 0;
}
