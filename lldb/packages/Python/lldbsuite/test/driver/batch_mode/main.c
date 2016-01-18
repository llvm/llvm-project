#include <stdio.h>
#include <string.h>

int 
main (int argc, char **argv)
{
    if (argc >= 2 && strcmp (argv[1], "CRASH") == 0)
    {
      char *touch_me_not = (char *) 0;
      printf ("About to crash.\n");
      touch_me_not[0] = 'a';
    }
    printf ("Got there on time and it did not crash.\n");
    return 0;
}
