#include <stdio.h>

class ExcA {};
class ExcB {};
class ExcC {};
class ExcD {};
class ExcE {};
class ExcF {};
class ExcG {};

void foo(int a)
{
  if (a > 1)
    throw ExcG();
  else
    throw ExcC();
}

void never_throws() throw () {
  printf("this statement is cold and should be outlined\n");
}

int main(int argc, char **argv)
{
  for (unsigned i = 0; i < 1000; ++i) {
    try {
      if (argc == 2) {
        never_throws(); // should be cold
      }
      try {
        if (argc == 2) {
          never_throws(); // should be cold
        }
        throw ExcA();
      } catch (ExcA) {
        printf("catch 2\n");
        throw new int();
      }
    } catch (...) {
      printf("catch 1\n");
    }

    try {
      try {
        foo(argc);
      } catch (ExcC) {
        printf("caught ExcC\n");
      }
    } catch (ExcG) {
      printf("caught ExcG\n");
    }
  }

  return 0;
}
