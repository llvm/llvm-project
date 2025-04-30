#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *files[500];

int
main (int argc, char *argv[])
{
  FILE *fp;
  int i;

  for (i = 0; i < 500; i++) {
    files[i] = tempnam (NULL, "file");
    if (files[i] == NULL) {
      printf ("tempnam failed\n");
      exit (1);
    }
    printf ("file: %s\n", files[i]);
    fp = fopen (files[i], "w");
    fclose (fp);
  }

  for (i = 0; i < 500; i++)
    remove (files[i]);

  return 0;
}
