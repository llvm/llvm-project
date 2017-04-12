#include "other.h"

extern "C" void some_func();

void 
Other::DoSomething()
{
  some_func();
}

