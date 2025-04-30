#include <dlfcn.h>
#include <stdio.h>
#include <gnu/lib-names.h>

#define TEST_SO "$ORIGIN/tst-dlmopen1mod.so"

static int
do_test (void)
{
  void *h = dlopen (LIBC_SO, RTLD_LAZY|RTLD_NOLOAD);
  if (h == NULL)
    {
      printf ("cannot get handle for %s: %s\n", LIBC_SO, dlerror ());
      return 1;
    }

  Lmid_t ns = -10;
  if (dlinfo (h, RTLD_DI_LMID, &ns) != 0)
    {
      printf ("dlinfo for %s in %s failed: %s\n",
	      LIBC_SO, __func__, dlerror ());
      return 1;
    }

  if (ns != LM_ID_BASE)
    {
      printf ("namespace for %s not LM_ID_BASE\n", LIBC_SO);
      return 1;
    }

  if (dlclose (h) != 0)
    {
      printf ("dlclose for %s in %s failed: %s\n",
	      LIBC_SO, __func__, dlerror ());
      return 1;
    }

  h = dlmopen (LM_ID_NEWLM, TEST_SO, RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot get handle for %s: %s\n",
	      "tst-dlmopen1mod.so", dlerror ());
      return 1;
    }

  ns = -10;
  if (dlinfo (h, RTLD_DI_LMID, &ns) != 0)
    {
      printf ("dlinfo for %s in %s failed: %s\n",
	      "tst-dlmopen1mod.so", __func__, dlerror ());
      return 1;
    }

  if (ns == LM_ID_BASE)
    {
      printf ("namespace for %s is LM_ID_BASE\n", TEST_SO);
      return 1;
    }

  int (*fct) (Lmid_t) = dlsym (h, "foo");
  if (fct == NULL)
    {
      printf ("could not find %s: %s\n", "foo", dlerror ());
      return 1;
    }

  if (fct (ns) != 0)
    return 1;

  if (dlclose (h) != 0)
    {
      printf ("dlclose for %s in %s failed: %s\n",
	      TEST_SO, __func__, dlerror ());
      return 1;
    }

  return 0;
}

#include <support/test-driver.c>
