#include <string.h>
#include <unistd.h>

static void
init_0 (int argc __attribute__ ((unused)), char **argv)
{
  char *p = strrchr (argv [0], '/');

  if (p == NULL)
      return;

  p++;
  size_t len = strlen (p);
  write (STDOUT_FILENO, "init array in DSO: ", 19);
  write (STDOUT_FILENO, p, len);
  write (STDOUT_FILENO, "\n", 1);
}

void (*const init_array []) (int, char **)
     __attribute__ ((section (".init_array"), aligned (sizeof (void *)))) =
{
  &init_0,
};
