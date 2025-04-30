#include <unistd.h>

extern int dep3 (void);

static void
__attribute__ ((constructor))
init (void)
{
  write (1, "0", 1);
}

static void
__attribute__ ((destructor))
fini (void)
{
  write (1, "9\n", 2);
}

int
dep3 (void)
{
  return 42;
}
