#include <stdio.h>
#include <stdlib.h>

void Dtor() { printf("destructor\n"); }

int Ctor() {
  printf("constructor\n");
  atexit(Dtor);
  return 0;
}

#pragma section(".CRT$XIV", long, read)
__declspec(allocate(".CRT$XIV")) int (*i1)(void) = Ctor;