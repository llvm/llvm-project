#include <stdint.h>

__thread int var;

unsigned int
la_version (unsigned int v)
{
  return v;
}

void
la_activity (uintptr_t *cookie, unsigned int flag)
{
  ++var;
}
