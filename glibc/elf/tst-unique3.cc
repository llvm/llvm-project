#include "tst-unique3.h"

#include <cstdio>
#include "../dlfcn/dlfcn.h"

int t = S<char>::i;

int
main (void)
{
  std::printf ("%d %d\n", S<char>::i, t);
  int result = S<char>::i++ != 1 || t != 1;
  result |= in_lib ();
  void *d = dlopen ("$ORIGIN/tst-unique3lib2.so", RTLD_LAZY);
  int (*fp) ();
  if (d == NULL || (fp = (int(*)()) dlsym (d, "in_lib2")) == NULL)
    {
      std::printf ("failed to get symbol in_lib2\n");
      return 1;
    }
  result |= fp ();
  dlclose (d);
  return result;
}
