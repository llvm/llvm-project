#include <dlfcn.h>
#include <stdio.h>

int
main (void)
{
  if (dlopen ("nodlopenmod.so", RTLD_LAZY) != NULL)
    {
      puts ("opening \"nodlopenmod.so\" succeeded, FAIL");
      return 1;
    }

  puts ("opening \"nodlopenmod.so\" failed, OK");
  return 0;
}
