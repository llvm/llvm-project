#include <stdio.h>

extern void start_b2( void ) __attribute__((constructor));
extern void finish_b2( void ) __attribute__((destructor));

void
start_b2( void )
{
  printf( "start_b2\n" );
}

void
finish_b2( void )
{
  printf( "finish_b2\n" );
}
