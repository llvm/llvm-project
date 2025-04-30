#include <cstdio>
#include "tst-unique3.h"
template<typename T> int S<T>::i = 1;
static int i = S<char>::i;

int
in_lib (void)
{
  std::printf ("in_lib: %d %d\n", S<char>::i, i);
  return S<char>::i++ != 2 || i != 1;
}
