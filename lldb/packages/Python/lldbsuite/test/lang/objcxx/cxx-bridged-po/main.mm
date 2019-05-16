#include <CoreFoundation/CoreFoundation.h>

void stop() {}

int main(int argc, char **argv)
{
  int value = 42;
  CFNumberRef num;
  num = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &value);
  stop(); // break here
  return 0;
}
