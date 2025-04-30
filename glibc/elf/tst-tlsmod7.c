#include "tst-tls10.h"

__thread int dummy __attribute__((visibility ("hidden"))) = 12;
__thread struct A a1 = { 4, 5, 6 };
__thread struct A a2 = { 7, 8, 9 };
__thread struct A a3 __attribute__((tls_model("initial-exec")))
  = { 10, 11, 12 };
__thread struct A a4 __attribute__((tls_model("initial-exec")))
  = { 13, 14, 15 };
static __thread struct A local1 = { 16, 17, 18 };
static __thread struct A local2 __attribute__((tls_model("initial-exec")))
  = { 19, 20, 21 };

void
check1 (void)
{
  if (a1.a != 4 || a1.b != 5 || a1.c != 6)
    abort ();
  if (a2.a != 22 || a2.b != 23 || a2.c != 24)
    abort ();
  if (a3.a != 10 || a3.b != 11 || a3.c != 12)
    abort ();
  if (a4.a != 25 || a4.b != 26 || a4.c != 27)
    abort ();
  if (local1.a != 16 || local1.b != 17 || local1.c != 18)
    abort ();
  if (local2.a != 19 || local2.b != 20 || local2.c != 21)
    abort ();
}

struct A *
f1a (void)
{
  return &a1;
}

struct A *
f2a (void)
{
  return &a2;
}

struct A *
f3a (void)
{
  return &a3;
}

struct A *
f4a (void)
{
  return &a4;
}

struct A *
f5a (void)
{
  return &local1;
}

struct A *
f6a (void)
{
  return &local2;
}

int
f1b (void)
{
  return a1.a;
}

int
f2b (void)
{
  return a2.b;
}

int
f3b (void)
{
  return a3.c;
}

int
f4b (void)
{
  return a4.a;
}

int
f5b (void)
{
  return local1.b;
}

int
f6b (void)
{
  return local2.c;
}
