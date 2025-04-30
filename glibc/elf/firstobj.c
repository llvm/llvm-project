#include <errno.h>

extern int foo (void);

int
foo (void)
{
  errno = 0;
  return 0;
}
