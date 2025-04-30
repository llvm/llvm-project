#include "testobj.h"

int
obj1func1 (int a)
{
  return 42 + obj1func2 (a);
}
