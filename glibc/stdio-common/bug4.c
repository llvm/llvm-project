#include <stdio.h>
#include <unistd.h>
#include <string.h>

int stdio_block_read = 1, stdio_block_write = 1;

int
main (int argc, char *argv[])
{
  FILE *f;
  int i;
  char buffer[31];
  const char filename[] = OBJPFX "bug4.test";

  while ((i = getopt (argc, argv, "rw")) != -1)
    switch (i)
      {
      case 'r':
	stdio_block_read = 0;
	break;
      case 'w':
	stdio_block_write = 0;
	break;
      }

  f = fopen (filename, "w+");
  for (i = 0; i < 9000; ++i)
    putc('x', f);

  fseek (f, 8180L, 0);
  fwrite ("Where does this text come from?", 1, 31, f);
  fseek (f, 8180L, 0);
  fread (buffer, 1, 31, f);
  fwrite (buffer, 1, 31, stdout);
  fclose (f);
  remove (filename);

  if (!memcmp (buffer, "Where does this text come from?", 31))
    {
      puts ("\nTest succeeded.");
      return 0;
    }
  else
    {
      puts ("\nTest FAILED!");
      return 1;
    }
}
