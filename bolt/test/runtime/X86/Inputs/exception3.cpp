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

void filter_only(int a) throw (ExcA, ExcB, ExcC, ExcD, ExcE, ExcF) {
  foo(a);
}

int main(int argc, char **argv)
{
  asm volatile ("nop;nop;nop;nop;nop");
  try {
    try {
      asm volatile ("nop;nop;nop;nop;nop");
      throw ExcA();
    } catch (ExcA) {
      asm volatile ("nop;nop;nop;nop;nop");
      printf("catch 2\n");
      throw new int();
    }
  } catch (...) {
    asm volatile ("nop;nop;nop;nop;nop");
    printf("catch 1\n");
  }

  try {
    asm volatile ("nop;nop;nop;nop;nop");
    try {
      asm volatile ("nop;nop;nop;nop;nop");
      filter_only(argc);
    } catch (ExcC) {
      asm volatile ("nop;nop;nop;nop;nop");
      printf("caught ExcC\n");
    }
  } catch (ExcG) {
    asm volatile ("nop;nop;nop;nop;nop");
    printf("caught ExcG\n");
  }

  return 0;
}
