#include <dlfcn.h>
#include <stdio.h>

extern int found_in_mod1 (void);
int
found_in_mod1 (void)
{
  return 1;
}


extern int test_in_mod1 (int (*mainp)(int, char **));
int
test_in_mod1 (int (*mainp)(int, char **))
{
  int (*ifp) (void);
  void *p;
  int result = 0;

  /* Find function `main'.  */
  p = dlsym (RTLD_DEFAULT, "main");
  if (p == NULL)
    {
      printf ("%s: main not found\n", __FILE__);
      result = 1;
    }
  else if ((int (*)(int, char **))p != mainp)
    {
      printf ("%s: wrong address returned for main\n", __FILE__);
      result = 1;
    }
  else
    printf ("%s: main correctly found\n", __FILE__);

  ifp = dlsym (RTLD_DEFAULT, "found_in_mod1");
  if ((void *) ifp == NULL)
    {
      printf ("%s: found_in_mod1 not found\n", __FILE__);
      result = 1;
    }
  else if (ifp () != 1)
    {
      printf ("%s: wrong address returned for found_in_mod1\n", __FILE__);
      result = 1;
    }
  else
    printf ("%s: found_in_mod1 correctly found\n", __FILE__);

  ifp = dlsym (RTLD_DEFAULT, "found_in_mod2");
  if ((void *) ifp == NULL)
    {
      printf ("%s: found_in_mod2 not found\n", __FILE__);
      result = 1;
    }
  else if (ifp () != 2)
    {
      printf ("%s: wrong address returned for found_in_mod2\n", __FILE__);
      result = 1;
    }
  else
    printf ("%s: found_in_mod2 correctly found\n", __FILE__);

  return result;
}
