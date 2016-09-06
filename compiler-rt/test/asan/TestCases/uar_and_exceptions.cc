// Test that use-after-return works with exceptions.
// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_stack_use_after_return=1 %run %t

#include <stdio.h>

volatile char *g;

#ifndef FRAME_SIZE
# define FRAME_SIZE 100
#endif

#ifndef NUM_ITER
# define NUM_ITER 4000
#endif

#ifndef DO_THROW
# define DO_THROW 1
#endif

void Func(int depth) {
  char frame[FRAME_SIZE];
  g = &frame[0];
  if (depth)
    Func(depth - 1);
  else if (DO_THROW)
    throw 1;
}

int main(int argc, char **argv) {
  for (int i = 0; i < NUM_ITER; i++) {
    try {
      Func(argc * 100);
    } catch(...) {
    }
    if ((i % (NUM_ITER / 10)) == 0)
      fprintf(stderr, "done [%d]\n", i);
  }
  return 0;
}
