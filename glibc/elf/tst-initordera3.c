#include <stdio.h>

extern void start_a3( void ) __attribute__((constructor));
extern void finish_a3( void ) __attribute__((destructor));

void
start_a3( void )
{
  printf( "start_a3\n" );
}

void
finish_a3( void )
{
  printf( "finish_a3\n" );
}
