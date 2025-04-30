#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>


static int
do_test (void)
{
  static const char modname[] = "tst-tlsmod2.so";
  int result = 0;
  int *foop;
  int (*fp) (int, int *);
  void *h;

  h = dlopen (modname, RTLD_LAZY);
  if (h == NULL)
    {
      printf ("cannot open '%s': %s\n", modname, dlerror ());
      exit (1);
    }

  fp = dlsym (h, "in_dso");
  if (fp == NULL)
    {
      printf ("cannot get symbol 'in_dso': %s\n", dlerror ());
      exit (1);
    }

  size_t modid = -1;
  if (dlinfo (h, RTLD_DI_TLS_MODID, &modid))
    {
      printf ("dlinfo RTLD_DI_TLS_MODID failed: %s\n", dlerror ());
      result = 1;
    }
  else
    printf ("dlinfo says TLS module ID %Zu\n", modid);

  void *block;
  if (dlinfo (h, RTLD_DI_TLS_DATA, &block))
    {
      printf ("dlinfo RTLD_DI_TLS_DATA failed: %s\n", dlerror ());
      result = 1;
    }
  else if (block != NULL)
    {
      printf ("dlinfo RTLD_DI_TLS_DATA says %p but should be unallocated\n",
	      block);
      result = 1;
    }

  result |= fp (0, NULL);

  foop = dlsym (h, "foo");
  if (foop == NULL)
    {
      printf ("cannot get symbol 'foo' the second time: %s\n", dlerror ());
      exit (1);
    }
  if (*foop != 16)
    {
      puts ("foo != 16");
      result = 1;
    }

  /* Now the module's TLS block has been used and should appear.  */
  if (dlinfo (h, RTLD_DI_TLS_DATA, &block))
    {
      printf ("dlinfo RTLD_DI_TLS_DATA failed the second time: %s\n",
	      dlerror ());
      result = 1;
    }
  else if (block != foop)
    {
      printf ("dlinfo RTLD_DI_TLS_DATA says %p but should be %p\n",
	      block, foop);
      result = 1;
    }

  dlclose (h);

  return result;
}


#include <support/test-driver.c>
