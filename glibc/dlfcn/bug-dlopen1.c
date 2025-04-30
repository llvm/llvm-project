/* Test case by Bruno Haible.  It test whether the dynamic string
   token expansion can handle $ signs which do not start one of the
   recognized keywords.  */

#include <dlfcn.h>

int main (void)
{
  dlopen ("gnu-gettext-GetURL$1", RTLD_GLOBAL | RTLD_LAZY);
  dlopen ("gnu-gettext-GetURL${1", RTLD_GLOBAL | RTLD_LAZY);
  return 0;
}
