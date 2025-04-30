#include <stdio.h>

extern void start_a1( void ) __attribute__((constructor));
extern void finish_a1( void ) __attribute__((destructor));

void
start_a1( void )
{
  printf( "start_a1\n" );
}

void
finish_a1( void )
{
  printf( "finish_a1\n" );
}
