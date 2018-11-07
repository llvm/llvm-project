#include <unistd.h>

void func1() {}
void func2() {}

int main(void)
{
  func1();

  execlp("ls", "-l", "-h", (char*)0);

  func2();

  return 0;
}
