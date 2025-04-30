#include "../dlfcn/dlfcn.h"
#include <stdio.h>
#include <stdlib.h>

static int
do_test (void)
{
  int result = 0;

  /* This is a test for correct handling of dlopen failures for library that
     is loaded with RTLD_NODELETE flag.  The first dlopen should fail because
     of undefined symbols in shared library.  The second dlopen then verifies
     that library was properly unloaded.  */
  if (dlopen ("tst-nodelete-rtldmod.so", RTLD_NOW | RTLD_NODELETE) != NULL
      || dlopen ("tst-nodelete-rtldmod.so", RTLD_LAZY | RTLD_NOLOAD) != NULL)
    {
      printf ("RTLD_NODELETE test failed\n");
      result = 1;
    }

  /* This is a test for correct handling of dlopen failures for library that
     is linked with '-z nodelete' option and hence has DF_1_NODELETE flag.
     The first dlopen should fail because of undefined symbols in shared
     library.  The second dlopen then verifies that library was properly
     unloaded.  */
  if (dlopen ("tst-nodelete-zmod.so", RTLD_NOW) != NULL
      || dlopen ("tst-nodelete-zmod.so", RTLD_LAZY | RTLD_NOLOAD) != NULL)
    {
      printf ("-z nodelete test failed\n");
      result = 1;
    }

   /* This is a test for correct handling of dlopen failures for library
     with unique symbols.  The first dlopen should fail because of undefined
     symbols in shared library.  The second dlopen then verifies that library
     was properly unloaded.  */
  if (dlopen ("tst-nodelete-uniquemod.so", RTLD_NOW) != NULL
      || dlopen ("tst-nodelete-uniquemod.so", RTLD_LAZY | RTLD_NOLOAD) != NULL)
    {
      printf ("Unique symbols test failed\n");
      result = 1;
    }

  if (result == 0)
    printf ("SUCCESS\n");

  return result;
}

#include <support/test-driver.c>
