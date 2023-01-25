#include <stdio.h>
#include <stdlib.h>

int func(long long Input);

int main(int argc, char *argv[]) {
  int arg = atoi(argv[1]);
  printf("%d\n", func(arg));
  return 0;
}
