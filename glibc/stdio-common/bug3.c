#include <stdio.h>
#include <string.h>

int
main (void)
{
  FILE *f;
  int i;
  const char filename[] = OBJPFX "bug3.test";

  f = fopen(filename, "w+");
  for (i=0; i<9000; i++)
    putc ('x', f);
  fseek (f, 8180L, 0);
  fwrite ("Where does this text go?", 1, 24, f);
  fflush (f);

  rewind (f);
  for (i=0; i<9000; i++)
    {
      int j;

      if ((j = getc(f)) != 'x')
	{
	  if (i != 8180)
	    {
	      printf ("Test FAILED!");
	      return 1;
	    }
	  else
	    {
	      char buf[25];

	      buf[0] = j;
	      fread (buf + 1, 1, 23, f);
	      buf[24] = '\0';
	      if (strcmp (buf, "Where does this text go?") != 0)
		{
		  printf ("%s\nTest FAILED!\n", buf);
		  return 1;
		}
	      i += 23;
	    }
	}
    }

  fclose(f);
  remove(filename);

  puts ("Test succeeded.");

  return 0;
}
