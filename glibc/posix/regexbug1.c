#include <error.h>
#include <sys/types.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  regex_t re;
  regmatch_t ma[2];
  int reerr;
  int res = 0;

  re_set_syntax (RE_DEBUG);
  reerr = regcomp (&re, "0*[0-9][0-9]", 0);
  if (reerr != 0)
    {
      char buf[100];
      regerror (reerr, &re, buf, sizeof buf);
      error (EXIT_FAILURE, 0, "%s", buf);
    }

  if (regexec (&re, "002", 2, ma, 0) != 0)
    {
      error (0, 0, "\"0*[0-9][0-9]\" does not match \"002\"");
      res = 1;
    }
  puts ("Succesful match with \"0*[0-9][0-9]\"");

  regfree (&re);

  reerr = regcomp (&re, "[0a]*[0-9][0-9]", 0);
  if (reerr != 0)
    {
      char buf[100];
      regerror (reerr, &re, buf, sizeof buf);
      error (EXIT_FAILURE, 0, "%s", buf);
    }

  if (regexec (&re, "002", 2, ma, 0) != 0)
    {
      error (0, 0, "\"[0a]*[0-9][0-9]\" does not match \"002\"");
      res = 1;
    }
  puts ("Succesful match with \"[0a]*[0-9][0-9]\"");

  regfree (&re);

  return res;
}
