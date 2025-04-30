#include <dlfcn.h>
#include <libintl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAPS ((struct link_map *) _r_debug.r_map)

static int
check_loaded_objects (const char **loaded)
{
  struct link_map *lm;
  int n;
  int *found = NULL;
  int errors = 0;

  for (n = 0; loaded[n]; n++)
    /* NOTHING */;

  if (n)
    {
      found = (int *) alloca (sizeof (int) * n);
      memset (found, 0, sizeof (int) * n);
    }

  printf("   Name\n");
  printf(" --------------------------------------------------------\n");
  for (lm = MAPS; lm; lm = lm->l_next)
    {
      if (lm->l_name && lm->l_name[0])
	printf(" %s, count = %d\n", lm->l_name, (int) lm->l_direct_opencount);
      if (lm->l_type == lt_loaded && lm->l_name)
	{
	  int match = 0;
	  for (n = 0; loaded[n] != NULL; n++)
	    {
	      if (strcmp (basename (loaded[n]), basename (lm->l_name)) == 0)
	        {
		  found[n] = 1;
		  match = 1;
		  break;
		}
	    }

	  if (match == 0)
	    {
	      ++errors;
	      printf ("ERRORS: %s is not unloaded\n", lm->l_name);
	    }
	}
    }

  for (n = 0; loaded[n] != NULL; n++)
    {
      if (found[n] == 0)
        {
	  ++errors;
	  printf ("ERRORS: %s is not loaded\n", loaded[n]);
	}
    }

  return errors;
}

static int
load_dso (const char **loading, int undef, int flag)
{
  void *obj;
  const char *loaded[] = { NULL, NULL, NULL, NULL };
  int errors = 0;
  const char *errstring;

  printf ("\nThis is what is in memory now:\n");
  errors += check_loaded_objects (loaded);

  printf ("Loading shared object %s: %s\n", loading[0],
	 flag == RTLD_LAZY ? "RTLD_LAZY" : "RTLD_NOW");
  obj = dlopen (loading[0], flag);
  if (obj == NULL)
    {
      if (flag == RTLD_LAZY)
	{
	  ++errors;
	  printf ("ERRORS: dlopen shouldn't fail for RTLD_LAZY\n");
	}

      errstring = dlerror ();
      if (strstr (errstring, "undefined symbol") == 0
	  || strstr (errstring, "circlemod2_undefined") == 0)
	{
	  ++errors;
	  printf ("ERRORS: dlopen: `%s': Invalid error string\n",
		  errstring);
	}
      else
	printf ("dlopen: %s\n", errstring);
    }
  else
    {
      if (undef && flag == RTLD_NOW)
	{
	  ++errors;
	  printf ("ERRORS: dlopen shouldn't work for RTLD_NOW\n");
	}

      if (!undef)
	{
	  int (*func) (void);

	  func = dlsym (obj, "circlemod1");
	  if (func == NULL)
	    {
	      ++errors;
	      printf ("ERRORS: cannot get address of \"circlemod1\": %s\n",
		      dlerror ());
	    }
	  else if (func () != 3)
	    {
	      ++errors;
	      printf ("ERRORS: function \"circlemod1\" returned wrong result\n");
	    }
	}

      loaded[0] = loading[0];
      loaded[1] = loading[1];
      loaded[2] = loading[2];
    }
  errors += check_loaded_objects (loaded);

  if (obj)
    {
      printf ("UnLoading shared object %s\n", loading[0]);
      dlclose (obj);
      loaded[0] = NULL;
      loaded[1] = NULL;
      loaded[2] = NULL;
      errors += check_loaded_objects (loaded);
    }

  return errors;
}

int
main (void)
{
  int errors = 0;
  const char *loading[3];

  loading[0] = "circlemod1a.so";
  loading[1] = "circlemod2a.so";
  loading[2] = "circlemod3a.so";
  errors += load_dso (loading, 0, RTLD_LAZY);
  errors += load_dso (loading, 0, RTLD_NOW);

  loading[0] = "circlemod1.so";
  loading[1] = "circlemod2.so";
  loading[2] = "circlemod3.so";
  errors += load_dso (loading, 1, RTLD_LAZY);
  errors += load_dso (loading, 1, RTLD_NOW);

  if (errors != 0)
    printf ("%d errors found\n", errors);

  return errors;
}
