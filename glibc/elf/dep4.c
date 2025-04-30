#include <unistd.h>

extern int dep3 (void);
extern int dep4 (void);

static void
__attribute__ ((constructor))
init (void)
{
  write (1, "1", 1);
}

static void
__attribute__ ((destructor))
fini (void)
{
  write (1, "8", 1);
}

int
dep4 (void)
{
  return dep3 ();
}
