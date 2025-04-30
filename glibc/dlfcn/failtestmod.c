#include <dlfcn.h>
#include <stdio.h>


extern void constr (void) __attribute__ ((__constructor__));
void
__attribute__ ((__constructor__))
constr (void)
{
  void *handle;

  /* Open the library.  */
  handle = dlopen (NULL, RTLD_NOW);
  if (handle == NULL)
    {
      puts ("Cannot get handle to own object");
      return;
    }

  /* Get a symbol.  */
  dlsym (handle, "main");
  puts ("called dlsym() to get main");

  dlclose (handle);
}
