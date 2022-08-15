#include <stdio.h>

int Ctor() {
  printf("constructor\n");
  return 0;
}

#pragma section(".CRT$XIV", long, read)
__declspec(allocate(".CRT$XIV")) int (*i1)(void) = Ctor;
