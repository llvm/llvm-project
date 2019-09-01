#include "dylib.h"
#include <stdio.h>

int
doSomething()
{
  // Set a breakpoint here.
  if (&absent_weak_int != NULL)
    printf("In absent_weak_int: %d\n", absent_weak_int);
  if (absent_weak_function != NULL)
    printf("In absent_weak_func: %p\n", absent_weak_function);
  if (&present_weak_int != NULL)
    printf("In present_weak_int: %d\n", present_weak_int);
  if (present_weak_function != NULL)
    printf("In present_weak_func: %p\n", present_weak_function);

}

int
main()
{
  return doSomething();
}
