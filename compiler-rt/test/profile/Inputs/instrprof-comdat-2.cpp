#include "instrprof-comdat.h"

int bar(int I) {

  FOO<long> Foo;
  FOO<int> Foo2;

  if (I > 5)
    return (int)Foo.DoIt(10);
  else
    return (int)Foo2.DoIt(I);
}
