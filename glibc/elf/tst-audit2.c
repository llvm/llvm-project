/* Test case for early TLS initialization in dynamic linker.  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define MAGIC1 0xabcdef72
#define MAGIC2 0xd8675309
static __thread unsigned int magic[] = { MAGIC1, MAGIC2 };
static __thread int calloc_called;

#undef calloc

/* This calloc definition will be called by the dynamic linker itself.
   We test that interposed calloc is called by the dynamic loader, and
   that TLS is fully initialized by then.  */

void *
calloc (size_t n, size_t m)
{
  if (!calloc_called)
    {
      /* Allow our calloc to be called more than once.  */
      calloc_called = 1;
      if (magic[0] != MAGIC1 || magic[1] != MAGIC2)
	{
	  printf ("{%x, %x} != {%x, %x}\n",
		  magic[0], magic[1], MAGIC1, MAGIC2);
	  abort ();
	}
      magic[0] = MAGIC2;
      magic[1] = MAGIC1;
    }

  n *= m;
  void *ptr = malloc (n);
  if (ptr != NULL)
    memset (ptr, '\0', n);
  return ptr;
}

static int
do_test (void)
{
  /* Make sure that our calloc is called from the dynamic linker at least
     once.  */
  void *h = dlopen("$ORIGIN/tst-auditmod9b.so", RTLD_LAZY);
  if (h != NULL)
    dlclose (h);
  if (magic[1] != MAGIC1 || magic[0] != MAGIC2)
    {
      printf ("{%x, %x} != {%x, %x}\n", magic[0], magic[1], MAGIC2, MAGIC1);
      return 1;
    }

  return 0;
}

#include <support/test-driver.c>
