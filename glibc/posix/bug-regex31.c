#include <mcheck.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

int
main (void)
{
  mtrace ();

  int res = 0;
  char *buf = NULL;
  size_t len = 0;
  while (! feof (stdin))
    {
      ssize_t n = getline (&buf, &len, stdin);
      if (n <= 0)
	break;
      if (buf[n - 1] == '\n')
	buf[n - 1] = '\0';

      regex_t regex;
      int rc = regcomp (&regex, buf, REG_EXTENDED);
      if (rc != 0)
	printf ("%s: Error %d (expected)\n", buf, rc);
      else
	{
	  printf ("%s: succeeded !\n", buf);
	  res = 1;
	}
    }

  free (buf);

  return res;
}
