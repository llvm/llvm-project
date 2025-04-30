#include <cstdio>
#include "tst-unique3.h"

template<typename T> int S<T>::i;

extern "C"
int
in_lib2 ()
{
  std::printf ("in_lib2: %d\n", S<char>::i);
  return S<char>::i != 3;
}
