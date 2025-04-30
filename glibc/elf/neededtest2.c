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

int
main (void)
{
  void *obj2;
  void *obj3[2];
  const char *loaded[] = { NULL, NULL, NULL, NULL };
  int errors = 0;

  printf ("\nThis is what is in memory now:\n");
  errors += check_loaded_objects (loaded);
  printf ("\nLoading shared object neededobj2.so\n");
  obj2 = dlopen ("neededobj2.so", RTLD_LAZY);
  if (obj2 == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  loaded[0] = "neededobj1.so";
  loaded[1] = "neededobj2.so";
  errors += check_loaded_objects (loaded);
  printf ("\nLoading shared object neededobj3.so\n");
  obj3[0] = dlopen( "neededobj3.so", RTLD_LAZY);
  if (obj3[0] == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  loaded[2] = "neededobj3.so";
  errors += check_loaded_objects (loaded);
  printf ("\nNow loading shared object neededobj3.so again\n");
  obj3[1] = dlopen ("neededobj3.so", RTLD_LAZY);
  if (obj3[1] == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  errors += check_loaded_objects (loaded);
  printf ("\nClosing neededobj3.so once\n");
  dlclose (obj3[0]);
  errors += check_loaded_objects (loaded);
  printf ("\nClosing neededobj2.so\n");
  dlclose (obj2);
  errors += check_loaded_objects (loaded);
  printf ("\nClosing neededobj3.so for the second time\n");
  dlclose (obj3[1]);
  loaded[0] = NULL;
  loaded[1] = NULL;
  loaded[2] = NULL;
  errors += check_loaded_objects (loaded);
  if (errors != 0)
    printf ("%d errors found\n", errors);
  return errors;
}
