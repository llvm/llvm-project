#include <stdint.h>
#include <stdio.h>

#define AL 4096
struct foo
{
  int i;
} __attribute ((aligned (AL)));

static __thread struct foo f;
static struct foo g;


#ifndef FCT
# define FCT in_dso1
#endif


int
FCT (void)
{
  puts (__func__);

  int result = 0;

  int fail = (((uintptr_t) &f) & (AL - 1)) != 0;
  printf ("&f = %p %s\n", &f, fail ? "FAIL" : "OK");
  result |= fail;

  fail = (((uintptr_t) &g) & (AL - 1)) != 0;
  printf ("&g = %p %s\n", &g, fail ? "FAIL" : "OK");
  result |= fail;

  return result;
}
