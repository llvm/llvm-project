#include <stdio.h>

extern void start_a2( void ) __attribute__((constructor));
extern void finish_a2( void ) __attribute__((destructor));

void
start_a2( void )
{
  printf( "start_a2\n" );
}

void
finish_a2( void )
{
  printf( "finish_a2\n" );
}
