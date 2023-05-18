#include "instrprof-comdat.h"
int g;
extern int bar(int);

int main() {

  FOO<int> Foo;

  int Res = Foo.DoIt(10);

  if (Res > 10)
    g = bar(10);
  else
    g = bar(1) + bar(2);
  return 0;
}

