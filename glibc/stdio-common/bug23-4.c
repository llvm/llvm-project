#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#define LIMIT 1000000

int
main (void)
{
  struct rlimit lim;
  getrlimit (RLIMIT_STACK, &lim);
  lim.rlim_cur = 1048576;
  setrlimit (RLIMIT_STACK, &lim);
  char *fmtstr = malloc (4 * LIMIT + 1);
  if (fmtstr == NULL)
    abort ();
  char *output = malloc (LIMIT + 1);
  if (output == NULL)
    abort ();
  for (size_t i = 0; i < LIMIT; i++)
    memcpy (fmtstr + 4 * i, "%1$d", 4);
  fmtstr[4 * LIMIT] = '\0';
  int ret = snprintf (output, LIMIT + 1, fmtstr, 0);
  if (ret != LIMIT)
    abort ();
  for (size_t i = 0; i < LIMIT; i++)
    if (output[i] != '0')
      abort ();
  return 0;
}
