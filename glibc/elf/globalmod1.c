#include <dlfcn.h>
#include <stdio.h>

extern int test (void);

int
test (void)
{
  (void) dlopen ("reldepmod4.so", RTLD_LAZY | RTLD_GLOBAL);
  if (dlsym (RTLD_DEFAULT, "call_me") != NULL)
    {
      puts ("found \"call_me\"");
      return 0;
    }
  puts ("didn't find \"call_me\"");
  return 1;
}
