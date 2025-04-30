#include <dlfcn.h>

extern int successful_rtld_next_test (void);
extern void *failing_rtld_next_use (void);

int nextmod1_dummy_var;

int
successful_rtld_next_test (void)
{
  int (*fp) (void);

  /* Get the next function... */
  fp = (int (*) (void)) dlsym (RTLD_NEXT, __FUNCTION__);

  /* ...and simply call it.  */
  return fp ();
}


void *
failing_rtld_next_use (void)
{
  void *ret = dlsym (RTLD_NEXT, __FUNCTION__);

  /* Ensure we are not tail call optimized, because then RTLD_NEXT
     might return this function.  */
  ++nextmod1_dummy_var;
  return ret;
}
