#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main (void)
{
  char *bp;
  size_t size;
  FILE *stream;
  int lose = 0;

  stream = open_memstream (&bp, &size);
  fprintf (stream, "hello");
  fflush (stream);
  printf ("buf = %s, size = %Zu\n", bp, size);
  lose |= size != 5;
  lose |= strncmp (bp, "hello", size);
  fprintf (stream, ", world");
  fclose (stream);
  printf ("buf = %s, size = %Zu\n", bp, size);
  lose |= size != 12;
  lose |= strncmp (bp, "hello, world", 12);

  puts (lose ? "Test FAILED!" : "Test succeeded.");

  free (bp);

  return lose;
}
