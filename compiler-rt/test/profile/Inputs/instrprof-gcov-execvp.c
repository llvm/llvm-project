#include <unistd.h>

void func1() {}
void func2() {}

int main(void)
{
  char *const args[] = {"-l", "-h", (char*)0};

  func1();

  execvp("ls", args);

  func2();

  return 0;
}
