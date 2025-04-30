#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

static int
do_test (void)
{
  printf( "main\n" );
  exit(EXIT_SUCCESS);
}

#include <support/test-driver.c>
