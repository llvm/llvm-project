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

extern void c_function (void);
extern char *dirname (const char *__filename);

int
main (int argc, char **argv)
{
  void *obj;
  const char *loaded[] = { NULL, NULL, NULL};
  int errors = 0;
  void (*f) (void);
  const char *dir = dirname (argv [0]);
  char *oldfilename;
  char *newfilename;

  c_function ();

  printf ("\nThis is what is in memory now:\n");
  errors += check_loaded_objects (loaded);

  printf( "Loading shared object neededobj6.so\n");
  obj = dlopen( "neededobj6.so", RTLD_LAZY);
  if (obj == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  f = dlsym (obj, "a2_function");
  if (f == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  f ();
  loaded[0] = "neededobj5.so";
  loaded[1] = "neededobj6.so";
  errors += check_loaded_objects (loaded);

  printf ("Closing neededobj6.so\n");
  dlclose (obj);
  loaded[0] = NULL;
  errors += check_loaded_objects (loaded);

  printf ("Rename neededobj5.so\n");
  oldfilename = alloca (strlen (dir) + 1 + sizeof ("neededobj5.so"));
  strcpy (oldfilename, dir);
  strcat (oldfilename, "/");
  strcat (oldfilename, "neededobj5.so");
  newfilename = alloca (strlen (oldfilename) + sizeof (".renamed"));
  strcpy (newfilename, oldfilename);
  strcat (newfilename, ".renamed");
  if (rename (oldfilename, newfilename))
    {
      perror ("rename");
      exit (1);
    }

  printf( "Loading shared object neededobj6.so\n");
  obj = dlopen( "neededobj6.so", RTLD_LAZY);
  if (obj == NULL)
    printf ("%s\n", dlerror ());
  else
    {
      printf ("neededobj6.so should fail to load\n");
      exit (1);
    }

  printf( "Loading shared object neededobj1.so\n");
  obj = dlopen( "neededobj1.so", RTLD_LAZY);
  if (obj == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  errors += check_loaded_objects (loaded);
  f = dlsym (obj, "c_function");
  if (f == NULL)
    {
      printf ("%s\n", dlerror ());
      exit (1);
    }
  f ();

  printf ("Restore neededobj5.so\n");
  if (rename (newfilename, oldfilename))
    {
      perror ("rename");
      exit (1);
    }

  if (errors != 0)
    printf ("%d errors found\n", errors);
  return errors;
}
