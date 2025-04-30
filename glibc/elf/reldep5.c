#include <dlfcn.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  void *h1;
  void *h2;
  int (*fp) (void);

  mtrace ();

  /* Open the two objects.  */
  h1 = dlopen ("reldepmod5.so", RTLD_LAZY);
  if (h1 == NULL)
    {
      printf ("cannot open reldepmod5.so: %s\n", dlerror ());
      exit (1);
    }
  h2 = dlopen ("reldepmod6.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      printf ("cannot open reldepmod6.so: %s\n", dlerror ());
      exit (1);
    }

  /* Get the address of the variable in reldepmod1.so.  */
  fp = dlsym (h2, "bar");
  if (fp == NULL)
    {
      printf ("cannot get address of \"bar\": %s\n", dlerror ());
      exit (1);
    }

  /* Call the function.  */
  puts ("calling fp for the first time");
  if (fp () != 0)
    {
      puts ("function \"call_me\" returned wrong result");
      exit (1);
    }

  /* Now close the first object.  It must still be around since we have
     an implicit dependency.  */
  if (dlclose (h1) != 0)
    {
      printf ("closing h1 failed: %s\n", dlerror ());
      exit (1);
    }

  /* Calling the function must still work.  */
  puts ("calling fp for the second time");
  if (fp () != 0)
    {
      puts ("function \"call_me\" the second time returned wrong result");
      exit (1);
    }
  puts ("second call suceeded as well");

  /* Close the second object, we are done.  */
  if (dlclose (h2) != 0)
    {
      printf ("closing h2 failed: %s\n", dlerror ());
      exit (1);
    }

  return 0;
}
