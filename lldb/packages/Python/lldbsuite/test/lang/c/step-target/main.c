#include <stdio.h>

void
lotsOfArgs
(
  int firstArg,
  int secondArg,
  int thirdArg,
  int fourthArg
)
{
  printf ("First: %d Second: %d Third: %d Fourth: %d.\n",
          firstArg,
          secondArg,
          thirdArg,
          fourthArg);
}

int
modifyInt(int incoming)
{
  return incoming % 2;
}

int
main (int argc, char **argv)
{
  if (argc > 0)
    {
      int var_makes_block = argc + 1;
      printf ("Break here to try targetted stepping.\n");
      lotsOfArgs(var_makes_block,
                 modifyInt(20),
                 30,
                 modifyInt(40));
      printf ("Done calling lotsOfArgs.");
    }
  printf ("All done.\n");
  return 0;
}
