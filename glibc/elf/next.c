#include <stdio.h>


extern int successful_rtld_next_test (void);
extern void *failing_rtld_next_use (void);


static int
do_test (void)
{
  int result;
  void *addr;

  /* First try call a function which uses RTLD_NEXT and calls that
     function.  */
  result = successful_rtld_next_test ();
  if (result == 42)
    {
      puts ("RTLD_NEXT seems to work for existing functions");
      result = 0;
    }
  else
    {
      printf ("Heh?  `successful_rtld_next_test' returned %d\n", result);
      result = 1;
    }

  /* Next try a function which tries to get a function with RTLD_NEXT
     but that fails.  This dlsym() call should return a NULL pointer
     and do nothing else.  */
  addr = failing_rtld_next_use ();
  if (addr == NULL)
    puts ("dlsym returned NULL for non-existing function.  Good");
  else
    {
      puts ("dlsym found something !?");
      result = 1;
    }

  return result;
}

#include <support/test-driver.c>
