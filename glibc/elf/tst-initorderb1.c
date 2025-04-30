#include <stdio.h>

extern void start_b1( void ) __attribute__((constructor));
extern void finish_b1( void ) __attribute__((destructor));

void
start_b1( void )
{
  printf( "start_b1\n" );
}

void
finish_b1( void )
{
  printf( "finish_b1\n" );
}
