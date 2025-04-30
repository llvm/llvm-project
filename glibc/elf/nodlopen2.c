#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  if (dlopen ("nodlopenmod2.so", RTLD_LAZY) != NULL)
    {
      puts ("opening \"nodlopenmod2.so\" succeeded, FAIL");
      return 1;
    }

  puts ("opening \"nodlopenmod2.so\" failed, OK");
  return 0;
}
