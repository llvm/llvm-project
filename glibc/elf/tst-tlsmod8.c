#include "tst-tls10.h"

__thread long long dummy __attribute__((visibility ("hidden"))) = 12;
__thread struct A a2 = { 22, 23, 24 };
__thread struct A a4 __attribute__((tls_model("initial-exec")))
  = { 25, 26, 27 };
static __thread struct A local1 = { 28, 29, 30 };
static __thread struct A local2 __attribute__((tls_model("initial-exec")))
  = { 31, 32, 33 };

void
check2 (void)
{
  if (a2.a != 22 || a2.b != 23 || a2.c != 24)
    abort ();
  if (a4.a != 25 || a4.b != 26 || a4.c != 27)
    abort ();
  if (local1.a != 28 || local1.b != 29 || local1.c != 30)
    abort ();
  if (local2.a != 31 || local2.b != 32 || local2.c != 33)
    abort ();
}

struct A *
f7a (void)
{
  return &a2;
}

struct A *
f8a (void)
{
  return &a4;
}

struct A *
f9a (void)
{
  return &local1;
}

struct A *
f10a (void)
{
  return &local2;
}

int
f7b (void)
{
  return a2.b;
}

int
f8b (void)
{
  return a4.a;
}

int
f9b (void)
{
  return local1.b;
}

int
f10b (void)
{
  return local2.c;
}
