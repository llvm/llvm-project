#include <stdio.h>

static int __thread tbaz __attribute__ ((tls_model ("local-dynamic"))) = 42;

void
setter2 (int a)
{
  tbaz = a;
}

int
baz (void)
{
  printf ("&tbaz=%p\n", &tbaz);
  return tbaz;
}
