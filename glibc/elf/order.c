#include <unistd.h>

void init (void) __attribute__ ((constructor));
void
__attribute__ ((constructor))
init (void)
{
  write (1, "4", 1);
}

void fini (void) __attribute__ ((destructor));
void
__attribute__ ((destructor))
fini (void)
{
  write (1, "5", 1);
}

extern int dep1 (void);

int
main (void)
{
  return dep1 () != 42;
}
