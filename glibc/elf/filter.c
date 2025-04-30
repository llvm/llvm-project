#include <mcheck.h>
#include <stdio.h>
#include <string.h>

extern const char *foo (void);

int
main (void)
{
  const char *s;

  mtrace ();

  s = foo ();

  printf ("called `foo' from `%s'\n", s);

  return strcmp (s, "filtmod2.c");
}
