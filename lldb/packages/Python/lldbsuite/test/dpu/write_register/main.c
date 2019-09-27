#define TASKLETS_INITIALIZER TASKLET( main, 1024, 0)
#include <rt.h>

int main() {
  __asm__("add r0, r0, 1\n");
  return 0;
}
