#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <link.h>


static int
do_test (void)
{
  static const char modname[] = "tst-tlsmod3.so";
  int result = 0;
  int (*fp) (void);
  void *h;
  int i;
  int modid = -1;

  for (i = 0; i < 10; ++i)
    {
      h = dlopen (modname, RTLD_LAZY);
      if (h == NULL)
	{
	  printf ("cannot open '%s': %s\n", modname, dlerror ());
	  exit (1);
	}

      /* Dirty test code here: we peek into a private data structure.
	 We make sure that the module gets assigned the same ID every
	 time.  The value of the first round is used.  */
      if (modid == -1)
	modid = ((struct link_map *) h)->l_tls_modid;
      else if (((struct link_map *) h)->l_tls_modid != (size_t) modid)
	{
	  printf ("round %d: modid now %zu, initially %d\n",
		  i, ((struct link_map *) h)->l_tls_modid, modid);
	  result = 1;
	}

      fp = dlsym (h, "in_dso2");
      if (fp == NULL)
	{
	  printf ("cannot get symbol 'in_dso2': %s\n", dlerror ());
	  exit (1);
	}

      result |= fp ();

      dlclose (h);
    }

  return result;
}


#include <support/test-driver.c>
