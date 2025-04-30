/* setjmp vs alloca test case.  Exercised bug on sparc.  */

#include <stdio.h>
#include <setjmp.h>
#include <alloca.h>

static void
sub5 (jmp_buf buf)
{
  longjmp (buf, 1);
}

static void
test (int x)
{
  jmp_buf buf;
  char *volatile foo;
  int arr[100];

  arr[77] = x;
  if (setjmp (buf))
    {
      printf ("made it ok; %d\n", arr[77]);
      return;
    }

  foo = (char *) alloca (128);
  (void) foo;
  sub5 (buf);
}

int
main (void)
{
  int i;

  for (i = 123; i < 345; ++i)
    test (i);

  return 0;
}
