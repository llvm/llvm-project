// Make sure all printed values are the same and are updated after BOLT.

#include <stdio.h>

int main(int argc, char *argv[]);

unsigned long Global = (unsigned long)main + 0x7fffffff;

int main(int argc, char *argv[]) {

  unsigned long Local = (unsigned long)&main + 0x7fffffff;
  unsigned long Local2 = &main + 0x7fffffff;

  printf("Global = 0x%lx\n", Global);
  printf("Local = 0x%lx\n", Local);
  printf("Local2 = 0x%lx\n", Local2);

  return 0;
}
