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
  int *vp;

  mtrace ();

  /* Open the two objects.  */
  h1 = dlopen ("reldepmod1.so", RTLD_LAZY | RTLD_GLOBAL);
  if (h1 == NULL)
    {
      printf ("cannot open reldepmod1.so: %s\n", dlerror ());
      exit (1);
    }
  h2 = dlopen ("reldepmod4.so", RTLD_LAZY);
  if (h2 == NULL)
    {
      printf ("cannot open reldepmod4.so: %s\n", dlerror ());
      exit (1);
    }

  /* Get the address of the variable in reldepmod1.so.  */
  vp = dlsym (h1, "some_var");
  if (vp == NULL)
    {
      printf ("cannot get address of \"some_var\": %s\n", dlerror ());
      exit (1);
    }

  *vp = 42;

  /* Get the function `call_me' in the second object.  This has a
     dependency which is resolved by a definition in reldepmod1.so.  */
  fp = dlsym (h2, "call_me");
  if (fp == NULL)
    {
      printf ("cannot get address of \"call_me\": %s\n", dlerror ());
      exit (1);
    }

  /* Call the function.  */
  if (fp () != 0)
    {
      puts ("function \"call_me\" returned wrong result");
      exit (1);
    }

  /* Now close the first object.  If must still be around since we have
     an implicit dependency.  */
  if (dlclose (h1) != 0)
    {
      printf ("closing h1 failed: %s\n", dlerror ());
      exit (1);
    }

  /* Open the first object again.   */
  h1 = dlopen ("reldepmod1.so", RTLD_LAZY | RTLD_GLOBAL);
  if (h1 == NULL)
    {
      printf ("cannot open reldepmod1.so the second time: %s\n", dlerror ());
      exit (1);
    }

  /* Get the variable address again.  */
  vp = dlsym (h1, "some_var");
  if (vp == NULL)
    {
      printf ("cannot get address of \"some_var\" the second time: %s\n",
	      dlerror ());
      exit (1);
    }

  /* The variable now must have its originial value.  */
  if (*vp != 0)
    {
      puts ("variable \"some_var\" not reset");
      exit (1);
    }

  /* Close the first object again, we are done.  */
  if (dlclose (h1) != 0)
    {
      printf ("closing h1 failed: %s\n", dlerror ());
      exit (1);
    }
  if (dlclose (h2) != 0)
    {
      printf ("closing h2 failed: %s\n", dlerror ());
      exit (1);
    }

  return 0;
}
