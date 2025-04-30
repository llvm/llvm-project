#include <unistd.h>

extern int dep1 (void);
extern int dep2 (void);
extern int dep4 (void);

static void
__attribute__ ((constructor))
init (void)
{
  write (1, "3", 1);
}

static void
__attribute__ ((destructor))
fini (void)
{
  write (1, "6", 1);
}

int
dep1 (void)
{
  return dep4 () - dep2 ();
}
