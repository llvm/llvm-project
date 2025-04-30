#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void
__attribute ((constructor))
init (void)
{
  puts ("init DSO");

  static char str[] = "SOMETHING_NOBODY_USES=something_else";
  if (putenv (str) != 0)
    {
      puts ("putenv failed");
      _exit (1);
    }
}
