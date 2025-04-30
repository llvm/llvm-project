#include "tst-tls10.h"

__thread int dummy __attribute__((visibility ("hidden"))) = 12;
__thread struct A local = { 1, 2, 3 };

#define CHECK(N, S)					\
  p = f##N##a ();					\
  if (p->a != S || p->b != S + 1 || p->c != S + 2)	\
    abort ()

static int
do_test (void)
{
  struct A *p;
  if (local.a != 1 || local.b != 2 || local.c != 3)
    abort ();
  if (a1.a != 4 || a1.b != 5 || a1.c != 6)
    abort ();
  if (a2.a != 22 || a2.b != 23 || a2.c != 24)
    abort ();
  if (a3.a != 10 || a3.b != 11 || a3.c != 12)
    abort ();
  if (a4.a != 25 || a4.b != 26 || a4.c != 27)
    abort ();
  check1 ();
  check2 ();
  if (f1a () != &a1 || f2a () != &a2 || f3a () != &a3 || f4a () != &a4)
    abort ();
  CHECK (5, 16);
  CHECK (6, 19);
  if (f7a () != &a2 || f8a () != &a4)
    abort ();
  CHECK (9, 28);
  CHECK (10, 31);

  exit (0);
}

#include <support/test-driver.c>
