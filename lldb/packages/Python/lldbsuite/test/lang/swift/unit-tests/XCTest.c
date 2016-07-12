#include <dlfcn.h>
#include <stdio.h>

int main()
{
  void *test_case = dlopen("UnitTest.xctest/Contents/MacOS/test", RTLD_NOW);

  printf("%p\n", test_case); // Set breakpoint here

  return 0;
}
