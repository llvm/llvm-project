#include <unistd.h>

extern int dep2 (void);
extern int dep3 (void);
extern int dep4 (void);

static void
__attribute__ ((constructor))
init (void)
{
  write (1, "2", 1);
}

static void
__attribute__ ((destructor))
fini (void)
{
  write (1, "7", 1);
}

int
dep2 (void)
{
  return dep3 () - dep4 ();
}
