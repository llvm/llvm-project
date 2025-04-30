#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <error.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* How many load/unload operations do we do.  */
#define TEST_ROUNDS	1000


static struct
{
  /* Name of the module.  */
  const char *name;
  /* The handle.  */
  void *handle;
} testobjs[] =
{
  { "testobj1.so", NULL },
  { "testobj2.so", NULL },
  { "testobj3.so", NULL },
  { "testobj4.so", NULL },
  { "testobj5.so", NULL },
  { "testobj6.so", NULL },
};
#define NOBJS	(sizeof (testobjs) / sizeof (testobjs[0]))


static const struct
{
  /* Name of a function to call.  */
  const char *fname;
  /* Index in status and handle array.  */
  int index;
  /* Options while loading the module.  */
  int options;
} tests[] =
{
  { "obj1func2", 0, RTLD_LAZY },
  { "obj1func1", 0, RTLD_LAZY | RTLD_GLOBAL },
  { "obj1func1", 0, RTLD_NOW, },
  { "obj1func2", 0, RTLD_NOW | RTLD_GLOBAL },
  { "obj2func2", 1, RTLD_LAZY },
  { "obj2func1", 1, RTLD_LAZY | RTLD_GLOBAL, },
  { "obj2func1", 1, RTLD_NOW, },
  { "obj2func2", 1, RTLD_NOW | RTLD_GLOBAL },
  { "obj3func2", 2, RTLD_LAZY },
  { "obj3func1", 2, RTLD_LAZY | RTLD_GLOBAL },
  { "obj3func1", 2, RTLD_NOW },
  { "obj3func2", 2, RTLD_NOW | RTLD_GLOBAL },
  { "obj4func2", 3, RTLD_LAZY },
  { "obj4func1", 3, RTLD_LAZY | RTLD_GLOBAL },
  { "obj4func1", 3, RTLD_NOW },
  { "obj4func2", 3, RTLD_NOW | RTLD_GLOBAL },
  { "obj5func2", 4, RTLD_LAZY },
  { "obj5func1", 4, RTLD_LAZY | RTLD_GLOBAL },
  { "obj5func1", 4, RTLD_NOW },
  { "obj5func2", 4, RTLD_NOW | RTLD_GLOBAL },
  { "obj6func2", 5, RTLD_LAZY },
  { "obj6func1", 5, RTLD_LAZY | RTLD_GLOBAL },
  { "obj6func1", 5, RTLD_NOW },
  { "obj6func2", 5, RTLD_NOW | RTLD_GLOBAL },
};
#define NTESTS	(sizeof (tests) / sizeof (tests[0]))


#include <include/link.h>

#define MAPS ((struct link_map *) _r_debug.r_map)

#define OUT							\
  do								\
    {								\
      for (map = MAPS; map != NULL; map = map->l_next)		\
	if (map->l_type == lt_loaded)				\
	  printf ("name = \"%s\", direct_opencount = %d\n",	\
		  map->l_name, (int) map->l_direct_opencount);	\
      fflush (stdout);						\
    }								\
  while (0)


int
main (int argc, char *argv[])
{
  int debug = argc > 1 && argv[1][0] != '\0';
  int count = TEST_ROUNDS;
  int result = 0;
  struct link_map *map;

  mtrace ();

  /* Just a seed.  */
  srandom (TEST_ROUNDS);

  if (debug)
    {
      puts ("in the beginning");
      OUT;
    }

  while (count--)
    {
      int nr = random () % NTESTS;
      int index = tests[nr].index;

      printf ("%4d: %4d: ", count + 1, nr);
      fflush (stdout);

      if (testobjs[index].handle == NULL)
	{
	  int (*fct) (int);

	  /* Load the object.  */
	  testobjs[index].handle = dlopen (testobjs[index].name,
					   tests[nr].options);
	  if (testobjs[index].handle == NULL)
	    error (EXIT_FAILURE, 0, "cannot load `%s': %s",
		   testobjs[index].name, dlerror ());

	  /* Test the function call.  */
	  fct = dlsym (testobjs[index].handle, tests[nr].fname);
	  if (fct == NULL)
	    error (EXIT_FAILURE, 0,
		   "cannot get function `%s' from shared object `%s': %s",
		   tests[nr].fname, testobjs[index].name, dlerror ());

	  fct (10);

	  printf ("successfully loaded `%s', handle %p\n",
		  testobjs[index].name, testobjs[index].handle);
	}
      else
	{
	  if (dlclose (testobjs[index].handle) != 0)
	    {
	      printf ("failed to close %s\n", testobjs[index].name);
	      result = 1;
	    }
	  else
	    printf ("successfully unloaded `%s', handle %p\n",
		    testobjs[index].name, testobjs[index].handle);

	  testobjs[index].handle = NULL;

	  if (testobjs[0].handle == NULL
	      && testobjs[1].handle == NULL
	      && testobjs[5].handle == NULL)
	    {
	      /* In this case none of the objects above should be
		 present.  */
	      for (map = MAPS; map != NULL; map = map->l_next)
		if (map->l_type == lt_loaded
		    && (strstr (map->l_name, testobjs[0].name) != NULL
			|| strstr (map->l_name, testobjs[1].name) != NULL
			|| strstr (map->l_name, testobjs[5].name) != NULL))
		  {
		    printf ("`%s' is still loaded\n", map->l_name);
		    result = 1;
		  }
	    }
	}

      if (debug)
	OUT;
    }

  /* Unload all loaded modules.  */
  for (count = 0; count < (int) NOBJS; ++count)
    if (testobjs[count].handle != NULL)
      {
	printf ("\nclose: %s: l_initfini = %p, l_versions = %p\n",
		testobjs[count].name,
		((struct link_map *) testobjs[count].handle)->l_initfini,
		((struct link_map *) testobjs[count].handle)->l_versions);

	if (dlclose (testobjs[count].handle) != 0)
	  {
	    printf ("failed to close %s\n", testobjs[count].name);
	    result = 1;
	  }
      }

  /* Check whether all files are unloaded.  */
  for (map = MAPS; map != NULL; map = map->l_next)
    if (map->l_type == lt_loaded)
      {
	printf ("name = \"%s\", direct_opencount = %d\n",
		map->l_name, (int) map->l_direct_opencount);
	result = 1;
      }

  return result;
}


extern int foo (int a);
int
foo (int a)
{
  return a - 1;
}
