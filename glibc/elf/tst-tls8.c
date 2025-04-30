#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#include <link.h>


static int
do_test (void)
{
  static const char modname1[] = "$ORIGIN/tst-tlsmod3.so";
  static const char modname2[] = "$ORIGIN/tst-tlsmod4.so";
  int result = 0;
  int (*fp1) (void);
  int (*fp2) (int, int *);
  void *h1;
  void *h2;
  int i;
  size_t modid1 = (size_t) -1;
  size_t modid2 = (size_t) -1;
  int *bazp;

  for (i = 0; i < 10; ++i)
    {
      h1 = dlopen (modname1, RTLD_LAZY);
      if (h1 == NULL)
	{
	  printf ("cannot open '%s': %s\n", modname1, dlerror ());
	  exit (1);
	}

      /* Dirty test code here: we peek into a private data structure.
	 We make sure that the module gets assigned the same ID every
	 time.  The value of the first round is used.  */
      if (modid1 == (size_t) -1)
	modid1 = ((struct link_map *) h1)->l_tls_modid;
      else if (((struct link_map *) h1)->l_tls_modid != modid1)
	{
	  printf ("round %d: modid now %zd, initially %zd\n",
		  i, ((struct link_map *) h1)->l_tls_modid, modid1);
	  result = 1;
	}

      fp1 = dlsym (h1, "in_dso2");
      if (fp1 == NULL)
	{
	  printf ("cannot get symbol 'in_dso2' in %s\n", modname1);
	  exit (1);
	}

      result |= fp1 ();



      h2 = dlopen (modname2, RTLD_LAZY);
      if (h2 == NULL)
	{
	  printf ("cannot open '%s': %s\n", modname2, dlerror ());
	  exit (1);
	}

      /* Dirty test code here: we peek into a private data structure.
	 We make sure that the module gets assigned the same ID every
	 time.  The value of the first round is used.  */
      if (modid2 == (size_t) -1)
	modid2 = ((struct link_map *) h1)->l_tls_modid;
      else if (((struct link_map *) h1)->l_tls_modid != modid2)
	{
	  printf ("round %d: modid now %zd, initially %zd\n",
		  i, ((struct link_map *) h1)->l_tls_modid, modid2);
	  result = 1;
	}

      bazp = dlsym (h2, "baz");
      if (bazp == NULL)
	{
	  printf ("cannot get symbol 'baz' in %s\n", modname2);
	  exit (1);
	}

      *bazp = 42 + i;

      fp2 = dlsym (h2, "in_dso");
      if (fp2 == NULL)
	{
	  printf ("cannot get symbol 'in_dso' in %s\n", modname2);
	  exit (1);
	}

      result |= fp2 (42 + i, bazp);

      dlclose (h1);
      dlclose (h2);


      h1 = dlopen (modname1, RTLD_LAZY);
      if (h1 == NULL)
	{
	  printf ("cannot open '%s': %s\n", modname1, dlerror ());
	  exit (1);
	}

      /* Dirty test code here: we peek into a private data structure.
	 We make sure that the module gets assigned the same ID every
	 time.  The value of the first round is used.  */
      if (((struct link_map *) h1)->l_tls_modid != modid1)
	{
	  printf ("round %d: modid now %zd, initially %zd\n",
		  i, ((struct link_map *) h1)->l_tls_modid, modid1);
	  result = 1;
	}

      fp1 = dlsym (h1, "in_dso2");
      if (fp1 == NULL)
	{
	  printf ("cannot get symbol 'in_dso2' in %s\n", modname1);
	  exit (1);
	}

      result |= fp1 ();



      h2 = dlopen (modname2, RTLD_LAZY);
      if (h2 == NULL)
	{
	  printf ("cannot open '%s': %s\n", modname2, dlerror ());
	  exit (1);
	}

      /* Dirty test code here: we peek into a private data structure.
	 We make sure that the module gets assigned the same ID every
	 time.  The value of the first round is used.  */
      if (((struct link_map *) h1)->l_tls_modid != modid2)
	{
	  printf ("round %d: modid now %zd, initially %zd\n",
		  i, ((struct link_map *) h1)->l_tls_modid, modid2);
	  result = 1;
	}

      bazp = dlsym (h2, "baz");
      if (bazp == NULL)
	{
	  printf ("cannot get symbol 'baz' in %s\n", modname2);
	  exit (1);
	}

      *bazp = 62 + i;

      fp2 = dlsym (h2, "in_dso");
      if (fp2 == NULL)
	{
	  printf ("cannot get symbol 'in_dso' in %s\n", modname2);
	  exit (1);
	}

      result |= fp2 (62 + i, bazp);

      /* This time the dlclose calls are in reverse order.  */
      dlclose (h2);
      dlclose (h1);
    }

  return result;
}

#include <support/test-driver.c>
