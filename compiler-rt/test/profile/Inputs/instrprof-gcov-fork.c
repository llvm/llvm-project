#include <unistd.h>

void func1() {}
void func2() {}

int main(void)
{
  func1();

  fork();

  func2();

  return 0;
}
