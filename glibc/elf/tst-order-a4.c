#include <stdio.h>

extern void start_a4( void ) __attribute__((constructor));
extern void finish_a4( void ) __attribute__((destructor));

void
start_a4( void )
{
  printf( "start_a4\n" );
}

void
finish_a4( void )
{
  printf( "finish_a4\n" );
}
