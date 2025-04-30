#include <stdio.h>
#include <string.h>

int
main(int argc, char *argv[])
{
  char buf[100];
  int a, b;
  int status = 0;

  sscanf ("12ab", "%dab%n", &a, &b);
  sprintf (buf, "%d, %d", a, b);
  puts (buf);
  status |= strcmp (buf, "12, 4");

  sscanf ("12ab100", "%dab%n100", &a, &b);
  sprintf (buf, "%d, %d", a, b);
  puts (buf);
  status |= strcmp (buf, "12, 4");

  return status;
}
