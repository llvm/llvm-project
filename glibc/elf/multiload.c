#include <dlfcn.h>
#include <errno.h>
#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int
main (void)
{
  void *a;
  void *b;
  void *c;
  void *d;
  char *wd;
  char *base;
  char *buf;

  mtrace ();

  /* Change to the binary directory.  */
  if (chdir (OBJDIR) != 0)
    {
      printf ("cannot change to `%s': %m", OBJDIR);
      exit (EXIT_FAILURE);
    }

  wd = getcwd (NULL, 0);
  base = basename (wd);
  buf = alloca (strlen (wd) + strlen (base) + 5 + sizeof "testobj1.so");

  printf ("loading `%s'\n", "./testobj1.so");
  a = dlopen ("./testobj1.so", RTLD_NOW);
  if (a == NULL)
    {
      printf ("cannot load `./testobj1.so': %s\n", dlerror ());
      exit (EXIT_FAILURE);
    }

  stpcpy (stpcpy (stpcpy (buf, "../"), base), "/testobj1.so");
  printf ("loading `%s'\n", buf);
  b = dlopen (buf, RTLD_NOW);
  if (b == NULL)
    {
      printf ("cannot load `%s': %s\n", buf, dlerror ());
      exit (EXIT_FAILURE);
    }

  stpcpy (stpcpy (buf, wd), "/testobj1.so");
  printf ("loading `%s'\n", buf);
  c = dlopen (buf, RTLD_NOW);
  if (c == NULL)
    {
      printf ("cannot load `%s': %s\n", buf, dlerror ());
      exit (EXIT_FAILURE);
    }

  stpcpy (stpcpy (stpcpy (stpcpy (buf, wd), "/../"), base), "/testobj1.so");
  printf ("loading `%s'\n", buf);
  d = dlopen (buf, RTLD_NOW);
  if (d == NULL)
    {
      printf ("cannot load `%s': %s\n", buf, dlerror ());
      exit (EXIT_FAILURE);
    }

  if (a != b || b != c || c != d)
    {
      puts ("shared object loaded more than once");
      exit (EXIT_FAILURE);
    }

  if (dlclose (a) != 0)
    {
      puts ("closing `a' failed");
      exit (EXIT_FAILURE);
    }
  if (dlclose (b) != 0)
    {
      puts ("closing `a' failed");
      exit (EXIT_FAILURE);
    }
  if (dlclose (c) != 0)
    {
      puts ("closing `a' failed");
      exit (EXIT_FAILURE);
    }
  if (dlclose (d) != 0)
    {
      puts ("closing `a' failed");
      exit (EXIT_FAILURE);
    }

  free (wd);

  return 0;
}

extern int foo (int a);
int
foo (int a)
{
  return a;
}
