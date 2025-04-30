#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char *argv[])
{
    int point, x, y;

    point = x = y = -1;
    sscanf("0x10 10", "%x %x", &x, &y);
    printf("%d %d\n", x, y);
    if (x != 0x10 || y != 0x10)
      abort ();
    point = x = y = -1;
    sscanf("P012349876", "P%1d%4d%4d", &point, &x, &y);
    printf("%d %d %d\n", point, x, y);
    if (point != 0 || x != 1234 || y != 9876)
      abort ();
    point = x = y = -1;
    sscanf("P112349876", "P%1d%4d%4d", &point, &x, &y);
    printf("%d %d %d\n", point, x, y);
    if (point != 1 || x != 1234 || y != 9876)
      abort ();
  return 0;
}
