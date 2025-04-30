#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

int
main (int argc, char *argv[])
{
  int n;
  char *line = NULL;
  size_t len = 0;

  if (argc == 1)
    {
      static char *new_argv[] = { NULL, (char *) "/dev/stdin", NULL };
      argv = new_argv;
      argc = 2;
    }

  for (n = 1; n < argc; ++n)
    {
      FILE *fp = fopen (argv[n], "r");
      if (fp == NULL)
	continue;

      while (! feof (fp))
	{
	  ssize_t cnt = getline (&line, &len, fp);
	  char *runp;
	  if (cnt <= 0)
	    break;

	  runp = line;
	  do
	    {
	      if (runp[0] == '<' && runp[1] == 'U' && isxdigit (runp[2])
		  && isxdigit (runp[3]) && isxdigit (runp[4])
		  && isxdigit (runp[5]) && runp[6] == '>')
		{
		  unsigned int val = strtoul (runp + 2, NULL, 16);

		  //putchar ('<');
		  if (val < 128)
		    putchar (val);
		  else if (val < 0x800)
		    {
		      putchar (0xc0 | (val >> 6));
		      putchar (0x80 | (val & 0x3f));
		    }
		  else
		    {
		      putchar (0xe0 | (val >> 12));
		      putchar (0x80 | ((val >> 6) & 0x3f));
		      putchar (0x80 | (val & 0x3f));
		    }
		  //putchar ('>');
		  runp += 7;
		}
	      else
		putchar (*runp++);
	    }
	  while (runp < &line[cnt]);
	}

      fclose (fp);
    }

  return 0;
}
